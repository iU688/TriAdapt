"""Equal-budget source-only pilots for R4-E context variants."""

import argparse
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common.paper_pose_metrics import SourcePaperMetricAccumulator
from common.pw3d_pair_dataset import BONES, PW3DPairDataset
from common.r3_multisource_dataset import MPIWeakPairDataset
from source_pose_geometry import EVAL_JOINTS_14
from models.r4e_unified_graphmlp import R4EUnifiedGraphMLP


def parse_args():
    parser = argparse.ArgumentParser(description="R4-E equal-budget architecture pilot")
    parser.add_argument("--variant", choices=sorted(R4EUnifiedGraphMLP.VARIANTS), required=True)
    parser.add_argument("--mpi_train", required=True)
    parser.add_argument("--mpi_val", required=True)
    parser.add_argument("--pw3d_train", required=True)
    parser.add_argument("--pw3d_val", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--resume_self_checkpoint")
    parser.add_argument("--frames", type=int, default=243)
    parser.add_argument("--dct_keep", type=int, default=27)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--early_stop", type=int, default=0,
                        help="stop after this many epochs without an eligible source gain; 0 disables")
    parser.add_argument("--samples_per_epoch", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--mpi_ratio", type=float, default=0.70)
    parser.add_argument("--pw3d_ratio", type=float, default=0.30)
    parser.add_argument("--paper_loss_weight", type=float, default=0.5)
    parser.add_argument("--gate_l1_weight", type=float, default=0.01)
    parser.add_argument("--repair_l2_weight", type=float, default=0.01)
    parser.add_argument("--metric_tolerance", type=float, default=1.0)
    parser.add_argument("--min_mean_gain", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--channel", type=int, default=512)
    parser.add_argument("--d_hid", type=int, default=1024)
    parser.add_argument("--token_dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--n_joints", type=int, default=17)
    parser.add_argument("--drop_rate", type=float, default=0.1)
    parser.add_argument("--adapter_dim", type=int, default=32)
    parser.add_argument("--cross_residual_scale", type=float, default=0.25)
    parser.add_argument("--pair_gate_scale", type=float, default=0.25)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_backbone(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_backbone_state(state, strict=False)
    unexpected = [key for key in unexpected if not key.endswith(".adj")]
    if missing or unexpected:
        raise ValueError(f"backbone mismatch: missing={missing}, unexpected={unexpected}")


def load_hybrid_self(model, path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "r4e_metadata" not in checkpoint:
        raise ValueError("self warm-start must be an R4-E checkpoint")
    metadata = checkpoint["r4e_metadata"]
    if metadata.get("variant") != "self":
        raise ValueError("self warm-start must come from variant=self")
    expected_frames = int(model.reliability.lowpass.shape[0])
    if int(metadata.get("frames", -1)) != expected_frames:
        raise ValueError("self checkpoint uses a different temporal length")
    model.load_hybrid_self_state(checkpoint["state_dict"])


def root_relative(prediction, target_mm):
    target = target_mm / 1000.0
    prediction = prediction - prediction[:, :, 0:1]
    target = target - target[:, :, 0:1]
    return prediction, target


def root_mpjpe_loss(prediction, target_mm):
    prediction, target = root_relative(prediction, target_mm)
    return torch.linalg.vector_norm(prediction - target, dim=-1).mean()


def paper_aligned_bone_loss(prediction, target_mm):
    """Differentiable counterpart of the reported bone-normalized 14-joint error."""
    prediction, target = root_relative(prediction, target_mm)
    pred_lengths, target_lengths = [], []
    for parent, child in BONES:
        pred_lengths.append(torch.linalg.vector_norm(
            prediction[:, :, child] - prediction[:, :, parent], dim=-1
        ))
        target_lengths.append(torch.linalg.vector_norm(
            target[:, :, child] - target[:, :, parent], dim=-1
        ))
    pred_scale = torch.stack(pred_lengths, dim=-1).mean(-1).clamp_min(1e-6)
    target_scale = torch.stack(target_lengths, dim=-1).mean(-1).clamp_min(1e-6)
    normalized = prediction * (target_scale / pred_scale)[..., None, None]
    joints = torch.as_tensor(EVAL_JOINTS_14, dtype=torch.long, device=prediction.device)
    return torch.linalg.vector_norm(
        normalized.index_select(2, joints) - target.index_select(2, joints), dim=-1
    ).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = SourcePaperMetricAccumulator()
    gates, reliabilities = [], []
    for batch in tqdm(loader, desc=f"[R4-E {model.variant} val]"):
        prediction, diagnostics = model(
            batch["detector"].to(device), batch["anchors"].to(device),
            batch["pair_is_distinct"].to(device), return_diagnostics=True,
        )
        metrics.update(batch["sequence"], prediction.cpu().numpy() * 1000.0,
                       batch["target"].numpy())
        gates.append(diagnostics["gate"].cpu().numpy())
        center = diagnostics["reliability"][:, :, diagnostics["reliability"].shape[2] // 2]
        reliabilities.append(center.mean(-1).cpu().numpy())
    result = metrics.summarize()
    result["gate_mean"] = float(np.concatenate(gates).mean())
    result["reliability_mean"] = float(np.concatenate(reliabilities).mean())
    return result


def train_epoch(model, loader, optimizer, device, args, epoch):
    model.train()
    model.backbone.eval()
    losses, root_losses, paper_losses = [], [], []
    gates, self_gates, cross_gates, pair_gates, pair_gate_magnitudes = [], [], [], [], []
    counts = {"mpi": 0, "pw3d": 0}
    distinct = 0
    for batch in tqdm(loader, desc=f"[R4-E {args.variant} train {epoch}/{args.epochs}]"):
        optimizer.zero_grad()
        target = batch["target"].to(device)
        pair_mask = batch["pair_is_distinct"].to(device)
        prediction, diagnostics = model(
            batch["detector"].to(device), batch["anchors"].to(device),
            pair_mask, return_diagnostics=True,
        )
        root_loss = root_mpjpe_loss(prediction, target)
        paper_loss = paper_aligned_bone_loss(prediction, target)
        gate = diagnostics["gate"]
        loss = (root_loss + args.paper_loss_weight * paper_loss
                + args.gate_l1_weight * gate.abs().mean()
                + args.repair_l2_weight * model.repair.strength.square())
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite R4-E loss")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item() * 1000.0))
        root_losses.append(float(root_loss.item() * 1000.0))
        paper_losses.append(float(paper_loss.item() * 1000.0))
        gates.append(float(gate.detach().mean().item()))
        self_gates.append(float(diagnostics["self_gate"].detach().mean().item()))
        cross_gates.append(float(diagnostics["cross_gate"].detach().mean().item()))
        pair_gates.append(float(diagnostics["pair_gate"].detach().mean().item()))
        pair_gate_magnitudes.append(
            float(diagnostics["pair_gate"].detach().abs().mean().item())
        )
        distinct += int(pair_mask.sum().item())
        for source in batch["source"]:
            counts[str(source)] += 1
    return {
        "loss": float(np.mean(losses)), "root_loss": float(np.mean(root_losses)),
        "paper_loss": float(np.mean(paper_losses)), "gate": float(np.mean(gates)),
        "self_gate": float(np.mean(self_gates)), "cross_gate": float(np.mean(cross_gates)),
        "pair_gate": float(np.mean(pair_gates)),
        "pair_gate_abs": float(np.mean(pair_gate_magnitudes)),
        "distinct": distinct, **counts,
    }


def mean_metrics(mpi, pw3d):
    return {key: 0.5 * (mpi[key] + pw3d[key]) for key in ("PCK_rel", "AUC_rel")}


def main():
    args = parse_args()
    if args.frames != 243:
        raise ValueError("R4-E pilot is locked to 243 frames")
    if args.early_stop < 0:
        raise ValueError("early_stop must be non-negative")
    warmstart_variants = {"hybrid", "pair_geometry"}
    if args.variant in warmstart_variants and not args.resume_self_checkpoint:
        raise ValueError(f"{args.variant} requires --resume_self_checkpoint")
    if args.variant not in warmstart_variants and args.resume_self_checkpoint:
        raise ValueError("--resume_self_checkpoint requires hybrid or pair_geometry")
    ratios = np.asarray([args.mpi_ratio, args.pw3d_ratio], dtype=np.float64)
    if np.any(ratios <= 0.0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("source ratios must be positive and sum to 1")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpi_train = MPIWeakPairDataset(args.mpi_train, args.frames, augment_flip=True, seed=args.seed)
    pw3d_train = PW3DPairDataset(args.pw3d_train, args.frames, center_stride=1,
                                 augment_flip=True, seed=args.seed, include_single=False)
    mpi_val = MPIWeakPairDataset(args.mpi_val, args.frames, augment_flip=False, seed=args.seed)
    pw3d_val = PW3DPairDataset(args.pw3d_val, args.frames, center_stride=1)
    train_sets = [mpi_train, pw3d_train]
    combined = ConcatDataset(train_sets)
    weights = torch.cat([
        torch.full((len(dataset),), float(ratio / len(dataset)), dtype=torch.double)
        for dataset, ratio in zip(train_sets, ratios)
    ])
    sampler = WeightedRandomSampler(
        weights, args.samples_per_epoch, replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, drop_last=True)
    mpi_loader = DataLoader(mpi_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers)
    pw3d_loader = DataLoader(pw3d_val, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers)
    model = R4EUnifiedGraphMLP(args, args.variant).to(device)
    load_backbone(model, args.checkpoint, device)
    if args.variant in warmstart_variants:
        load_hybrid_self(model, args.resume_self_checkpoint, device)
    model.freeze_backbone()
    model.freeze_hybrid_base()
    optimizer = torch.optim.AdamW(model.adaptation_parameters(), lr=args.lr, weight_decay=1e-4)
    trainable = sum(parameter.numel() for parameter in model.adaptation_parameters())
    print("[Protocol] genuine MPI pairs + genuine 3DPW pairs; source-only dual validation")
    print("[Target access] MuPoTS=none, 3DPW test=none")
    print(f"[Variant] {args.variant} trainable={trainable}")
    if args.variant in warmstart_variants:
        print(f"[Frozen self] checkpoint={args.resume_self_checkpoint}")
    if args.variant == "hybrid":
        print(f"[Hybrid] cross_residual_scale={args.cross_residual_scale}")
    elif args.variant == "pair_geometry":
        print(f"[Pair geometry] pair_gate_scale={args.pair_gate_scale} "
              "companion_pose_features=False")
    print(f"[Data] mpi={len(mpi_train)} pw3d_genuine_pairs={len(pw3d_train)} "
          f"mpi_val={len(mpi_val)} pw3d_val={len(pw3d_val)}")
    print(f"[Sampling] samples={args.samples_per_epoch} ratios={ratios.tolist()} all_pairs_distinct=True")
    print(f"[Schedule] max_epochs={args.epochs} early_stop={args.early_stop}")
    mpi0, pw3d0 = evaluate(model, mpi_loader, device), evaluate(model, pw3d_loader, device)
    mean0 = mean_metrics(mpi0, pw3d0)
    print(f"[Epoch 0] mean={mean0['PCK_rel']:.2f}/{mean0['AUC_rel']:.2f} "
          f"MPI={mpi0['PCK_rel']:.2f}/{mpi0['AUC_rel']:.2f} "
          f"3DPW={pw3d0['PCK_rel']:.2f}/{pw3d0['AUC_rel']:.2f}")
    if args.eval_only:
        print("[Eval-only] no checkpoint was saved")
        return

    os.makedirs("checkpoint", exist_ok=True)
    os.makedirs("outputs/train_logs", exist_ok=True)
    stamp = datetime.now().strftime("%m%d_%H%M")
    log_path = os.path.join("outputs/train_logs", f"r4e_{args.tag}_{args.variant}_{stamp}.csv")
    best_path = os.path.join("checkpoint", f"best_r4e_{args.tag}_{args.variant}.pth")
    fields = ["epoch", "loss", "root_loss", "paper_loss", "gate", "self_gate",
              "cross_gate", "pair_gate", "pair_gate_abs", "repair_strength",
              "mpi_samples", "pw3d_samples", "distinct_samples", "mpi_pck", "mpi_auc",
              "pw3d_pck", "pw3d_auc", "mean_pck", "mean_auc", "eligible"]
    with open(log_path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()
    best_score, best_epoch, patience = mean0["PCK_rel"] + mean0["AUC_rel"], 0, 0
    for epoch in range(1, args.epochs + 1):
        for dataset in train_sets:
            dataset.set_epoch(epoch)
        train = train_epoch(model, train_loader, optimizer, device, args, epoch)
        mpi, pw3d = evaluate(model, mpi_loader, device), evaluate(model, pw3d_loader, device)
        mean = mean_metrics(mpi, pw3d)
        domains_preserved = (
            mpi["PCK_rel"] >= mpi0["PCK_rel"] - args.metric_tolerance
            and pw3d["PCK_rel"] >= pw3d0["PCK_rel"] - args.metric_tolerance
        )
        if args.variant in warmstart_variants:
            pck_gain = mean["PCK_rel"] - mean0["PCK_rel"]
            auc_gain = mean["AUC_rel"] - mean0["AUC_rel"]
            eligible = domains_preserved and (
                (pck_gain >= args.min_mean_gain and auc_gain >= -args.min_mean_gain)
                or (auc_gain >= args.min_mean_gain and pck_gain >= -args.min_mean_gain)
            )
        else:
            eligible = (
                domains_preserved
                and mean["PCK_rel"] >= mean0["PCK_rel"] + args.min_mean_gain
                and mean["AUC_rel"] >= mean0["AUC_rel"] + args.min_mean_gain
            )
        score = mean["PCK_rel"] + mean["AUC_rel"]
        improved = eligible and score > best_score
        if improved:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "r4e_metadata": vars(args),
                        "epoch0_mpi": mpi0, "epoch0_pw3d": pw3d0}, best_path)
            print(f"  [best eligible] saved {best_path}")
        else:
            patience += 1
        print(f"[Epoch {epoch}] loss={train['loss']:.1f} root={train['root_loss']:.1f} "
              f"paper={train['paper_loss']:.1f} mean={mean['PCK_rel']:.2f}/{mean['AUC_rel']:.2f} "
              f"MPI={mpi['PCK_rel']:.2f}/{mpi['AUC_rel']:.2f} "
              f"3DPW={pw3d['PCK_rel']:.2f}/{pw3d['AUC_rel']:.2f} "
              f"self_gate={train['self_gate']:.4f} cross_gate={train['cross_gate']:.4f} "
              f"pair_gate={train['pair_gate']:.4f} "
              f"pair_gate_abs={train['pair_gate_abs']:.4f}")
        row = {"epoch": epoch, "loss": train["loss"], "root_loss": train["root_loss"],
               "paper_loss": train["paper_loss"], "gate": train["gate"],
               "self_gate": train["self_gate"], "cross_gate": train["cross_gate"],
               "pair_gate": train["pair_gate"],
               "pair_gate_abs": train["pair_gate_abs"],
               "repair_strength": float(model.repair.strength.item()),
               "mpi_samples": train["mpi"], "pw3d_samples": train["pw3d"],
               "distinct_samples": train["distinct"], "mpi_pck": mpi["PCK_rel"],
               "mpi_auc": mpi["AUC_rel"], "pw3d_pck": pw3d["PCK_rel"],
               "pw3d_auc": pw3d["AUC_rel"], "mean_pck": mean["PCK_rel"],
               "mean_auc": mean["AUC_rel"], "eligible": eligible}
        with open(log_path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        if args.early_stop > 0 and patience >= args.early_stop:
            print(f"  [early stop] no eligible source improvement for {patience} epochs")
            break
    print(f"[Done] best_epoch={best_epoch} best_score={best_score:.2f}")
    print(f"[Log] {log_path}")
    if best_epoch == 0:
        print("[Gate] FAIL")


if __name__ == "__main__":
    main()
