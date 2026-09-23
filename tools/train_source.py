from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler


PROJECT = Path(__file__).resolve().parents[1]
RUNTIME = PROJECT / "runtime"
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(RUNTIME))

from common.pw3d_pair_dataset import PW3DPairDataset
from common.r3_multisource_dataset import MPIWeakPairDataset
from common.paper_pose_metrics import SourcePaperMetricAccumulator
from mupots_multimodule.corruptions import apply_frozen_source_corruption
from mupots_multimodule.model import MuPoTSTargetedGraphMLP
from mupots_multimodule.project_paths import m2_checkpoint, source_asset
from train_r4e_equal_pilot import (
    BONES,
    EVAL_JOINTS_14,
    paper_aligned_bone_loss,
    root_mpjpe_loss,
    root_relative,
)


def finite_masked_smooth_l1(prediction, target, confidence):
    """Smooth-L1 that never evaluates arithmetic on non-finite target entries."""
    finite = torch.isfinite(target).all(dim=-1, keepdim=True)
    mask = (confidence.unsqueeze(-1) > 0.0) & finite
    safe_prediction = torch.where(mask, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    per_coordinate = F.smooth_l1_loss(safe_prediction, safe_target, reduction="none")
    return per_coordinate.sum() / (mask.sum().clamp_min(1) * prediction.shape[-1])


def smooth_pck_loss(prediction, target_mm, threshold_m=0.15, temperature_m=0.02):
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
    aligned = prediction * (target_scale / pred_scale)[..., None, None]
    joints = torch.as_tensor(EVAL_JOINTS_14, device=prediction.device)
    error = torch.linalg.vector_norm(
        aligned.index_select(2, joints) - target.index_select(2, joints), dim=-1
    )
    return torch.sigmoid((error - threshold_m) / temperature_m).mean()


def parse_args():
    parser = argparse.ArgumentParser(description="Source-only MuPoTS-targeted module training")
    parser.add_argument("--mpi_train", required=True)
    parser.add_argument("--mpi_val", required=True)
    parser.add_argument("--pw3d_train", required=True)
    parser.add_argument("--pw3d_val", required=True)
    parser.add_argument("--m2_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), required=True)
    parser.add_argument("--arm", choices=("m1_m2", "m2_m3", "full"), default="full")
    parser.add_argument("--stage", choices=("pilot", "formal"), required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--samples_per_epoch", type=int)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmstart_debug")
    parser.add_argument("--gate_baseline_json")
    parser.add_argument("--formal_protocol")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_formal_protocol(args, epochs: int, samples_per_epoch: int, output: Path) -> dict:
    if args.stage != "formal":
        if args.formal_protocol:
            raise ValueError("formal_protocol is valid only for formal training")
        return {}
    if not args.formal_protocol:
        raise ValueError("formal training requires --formal_protocol")
    path = Path(args.formal_protocol).resolve()
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "locked_before_formal_training":
        raise ValueError("formal protocol is not locked")
    if args.seed not in protocol["seeds"] or args.arm not in protocol["trained_arms"]:
        raise ValueError("seed/arm is outside the formal protocol")
    budget = protocol["training_budget"]
    observed_budget = {
        "epochs_max": epochs,
        "samples_per_epoch": samples_per_epoch,
        "early_stop": 5,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
    }
    if observed_budget != budget:
        raise ValueError(f"formal budget mismatch: {observed_budget} != {budget}")
    expected_output = (PROJECT / protocol["output_root"] / args.arm / f"seed{args.seed}").resolve()
    if output != expected_output:
        raise ValueError(f"formal output mismatch: {output} != {expected_output}")
    path_bindings = {
        "mpi_train": Path(args.mpi_train).resolve(),
        "mpi_val": Path(args.mpi_val).resolve(),
        "pw3d_train": Path(args.pw3d_train).resolve(),
        "pw3d_val": Path(args.pw3d_val).resolve(),
    }
    for name, value in path_bindings.items():
        locked = protocol["source_assets"][name]
        expected_path = source_asset(name)
        if value != expected_path or sha256_file(value) != locked["sha256"].upper():
            raise ValueError(f"formal source asset mismatch: {name}")
    checkpoint = Path(args.m2_checkpoint).resolve()
    locked_checkpoint = protocol["m2_checkpoints"][str(args.seed)]
    if checkpoint != m2_checkpoint(args.seed):
        raise ValueError("formal M2 checkpoint path mismatch")
    if sha256_file(checkpoint) != locked_checkpoint["sha256"].upper():
        raise ValueError("formal M2 checkpoint hash mismatch")
    code_paths = {
        "train_source": Path(__file__).resolve(),
        "model": PROJECT / "src" / "mupots_multimodule" / "model.py",
        "corruptions": PROJECT / "src" / "mupots_multimodule" / "corruptions.py",
        "pw3d_dataset": RUNTIME / "common" / "pw3d_pair_dataset.py",
        "mpi_dataset": RUNTIME / "common" / "r3_multisource_dataset.py",
        "paper_metrics": RUNTIME / "common" / "paper_pose_metrics.py",
        "source_losses": RUNTIME / "train_r4e_equal_pilot.py",
        "m2_checkpoint_loader": (
            RUNTIME / "experiments" / "mechanism_controls" / "control_checkpoint.py"
        ),
    }
    relocation_path = PROJECT / "config" / "relocation_integrity.json"
    expected_code = protocol["code_sha256"]
    if relocation_path.is_file():
        relocation = json.loads(relocation_path.read_text(encoding="utf-8"))
        expected_code = relocation["active_code_sha256"]
    for name, code_path in code_paths.items():
        if sha256_file(code_path) != expected_code[name].upper():
            raise ValueError(f"formal code hash mismatch: {name}")
    if args.warmstart_debug or args.gate_baseline_json:
        raise ValueError("formal training forbids warm-start and external gate baselines")
    return protocol


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, device, corrupt: bool, seed: int):
    model.eval()
    metrics = SourcePaperMetricAccumulator()
    generator = torch.Generator().manual_seed(seed)
    corrections, refinements = [], []
    for batch in loader:
        detector = batch["detector"].to(device)
        if corrupt:
            detector = apply_frozen_source_corruption(detector.cpu(), generator, 1.0).to(device)
        prediction, diagnostics = model(
            detector,
            batch["anchors"].to(device),
            batch["pair_is_distinct"].to(device),
            return_diagnostics=True,
        )
        metrics.update(batch["sequence"], prediction.cpu().numpy() * 1000.0,
                       batch["target"].numpy())
        corrections.append(float(diagnostics["correction_2d_rms"].item()))
        refinements.append(float(diagnostics["refinement_3d_rms"].item()))
    result = metrics.summarize()
    result["correction_2d_rms"] = float(np.mean(corrections))
    result["refinement_3d_rms"] = float(np.mean(refinements))
    return result


def mean_metrics(left, right):
    return {key: 0.5 * (left[key] + right[key]) for key in ("PCK_rel", "AUC_rel", "MPJPE_rel")}


def evaluate_pair(model, mpi_loader, pw_loader, device, corrupt: bool, seed: int):
    mpi = evaluate(model, mpi_loader, device, corrupt, seed + 11)
    pw = evaluate(model, pw_loader, device, corrupt, seed + 29)
    return {"mpi": mpi, "pw3d": pw, "mean": mean_metrics(mpi, pw)}


def main():
    args = parse_args()
    if args.num_workers != 0:
        raise ValueError("V1 locks num_workers=0")
    defaults = {
        "pilot": {"epochs": 8, "samples": 20_000, "early_stop": 8},
        "formal": {"epochs": 30, "samples": 100_000, "early_stop": 5},
    }[args.stage]
    epochs = args.epochs or defaults["epochs"]
    samples_per_epoch = args.samples_per_epoch or defaults["samples"]
    output = Path(args.output_dir).resolve()
    output.relative_to(PROJECT)
    verify_formal_protocol(args, epochs, samples_per_epoch, output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage == "formal" and device.type != "cuda":
        raise RuntimeError("formal training requires CUDA")

    mpi_train = MPIWeakPairDataset(args.mpi_train, 243, augment_flip=True, seed=args.seed)
    pw_train = PW3DPairDataset(args.pw3d_train, 243, center_stride=1,
                              augment_flip=True, seed=args.seed)
    mpi_val = MPIWeakPairDataset(args.mpi_val, 243, augment_flip=False, seed=args.seed)
    pw_val = PW3DPairDataset(args.pw3d_val, 243, center_stride=1, augment_flip=False)
    combined = ConcatDataset([mpi_train, pw_train])
    weights = torch.cat((
        torch.full((len(mpi_train),), 0.70 / len(mpi_train), dtype=torch.double),
        torch.full((len(pw_train),), 0.30 / len(pw_train), dtype=torch.double),
    ))
    sampler = WeightedRandomSampler(weights, samples_per_epoch, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(combined, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, drop_last=True)
    mpi_loader = DataLoader(mpi_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pw_loader = DataLoader(pw_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MuPoTSTargetedGraphMLP(args.m2_checkpoint, device).to(device)
    model.configure_arm(args.arm)
    if int(model.base_metadata["seed"]) != args.seed:
        raise ValueError("M2 checkpoint seed must match training seed")
    if args.warmstart_debug:
        if args.stage != "pilot" or not args.gate_baseline_json:
            raise ValueError("debug warm-start is pilot-only and requires gate_baseline_json")
        payload = torch.load(args.warmstart_debug, map_location=device, weights_only=False)
        metadata = payload.get("metadata", {})
        if metadata.get("target_authorized") is not False or metadata.get("target_access") != "none":
            raise ValueError("warm-start must be explicitly source-only and target-unauthorized")
        model.load_state_dict(payload["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(model.adaptation_parameters(), lr=2e-4, weight_decay=1e-4)
    trainable = sum(p.numel() for p in model.adaptation_parameters())
    if trainable >= 20_000:
        raise ValueError(f"new-module budget exceeded: {trainable}")

    clean0 = evaluate_pair(model, mpi_loader, pw_loader, device, False, args.seed)
    corrupt0 = evaluate_pair(model, mpi_loader, pw_loader, device, True, args.seed)
    gate_clean0, gate_corrupt0 = clean0, corrupt0
    if args.gate_baseline_json:
        frozen_baseline = json.loads(Path(args.gate_baseline_json).read_text(encoding="utf-8"))
        if "eligibility_reference" in frozen_baseline:
            frozen_baseline = frozen_baseline["eligibility_reference"]
        gate_clean0, gate_corrupt0 = frozen_baseline["clean"], frozen_baseline["corrupt"]
    baseline = {
        "observed_start": {"clean": clean0, "corrupt": corrupt0},
        "eligibility_reference": {"clean": gate_clean0, "corrupt": gate_corrupt0},
    }
    (output / "epoch0.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")
    fields = ["epoch", "loss", "root", "bone", "soft_pck", "repair2d", "residual",
              "clean_pck", "clean_auc", "corrupt_pck", "corrupt_auc", "eligible"]
    log_path = output / "training.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    corruption_generator = torch.Generator().manual_seed(args.seed + 100_000)
    baseline_score = sum(clean0["mean"][key] + corrupt0["mean"][key]
                         for key in ("PCK_rel", "AUC_rel"))
    best_score, best_epoch = -float("inf"), 0
    best_progress_score, patience = baseline_score, 0
    best_path = output / "best_source_selected.pth"
    progress_path = output / "best_progress_only_not_target_authorized.pth"
    for epoch in range(1, epochs + 1):
        mpi_train.set_epoch(epoch)
        pw_train.set_epoch(epoch)
        model.train()
        totals = {key: [] for key in ("loss", "root", "bone", "soft_pck", "repair2d", "residual")}
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            detector = batch["detector"]
            corrupted = apply_frozen_source_corruption(detector, corruption_generator, 0.5).to(device)
            prediction, diagnostics = model(
                corrupted,
                batch["anchors"].to(device),
                batch["pair_is_distinct"].to(device),
                return_diagnostics=True,
            )
            target = batch["target"].to(device)
            root = root_mpjpe_loss(prediction, target)
            bone = paper_aligned_bone_loss(prediction, target)
            soft_pck = smooth_pck_loss(prediction, target)
            gt2d = batch["gt2d"].to(device)
            gt_confidence = batch["gt_confidence"].to(device)
            repair2d = finite_masked_smooth_l1(
                diagnostics["corrected_2d"], gt2d, gt_confidence
            )
            residual = diagnostics["correction_2d_rms"] + diagnostics["refinement_3d_rms"]
            repair_weight = 0.0 if model.disable_corrector else 1.0
            loss = root + 0.5 * bone + 0.05 * soft_pck + repair_weight * repair2d + 0.01 * residual
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            optimizer.step()
            for key, value in (("loss", loss), ("root", root), ("bone", bone),
                               ("soft_pck", soft_pck),
                               ("repair2d", repair2d), ("residual", residual)):
                totals[key].append(float(value.detach().item()))

        clean = evaluate_pair(model, mpi_loader, pw_loader, device, False, args.seed)
        corrupt = evaluate_pair(model, mpi_loader, pw_loader, device, True, args.seed)
        clean_preserved = (
            clean["mean"]["PCK_rel"] >= gate_clean0["mean"]["PCK_rel"] - 0.50
            and clean["mean"]["AUC_rel"] >= gate_clean0["mean"]["AUC_rel"] - 0.50
        )
        corrupt_improved = (
            corrupt["mean"]["PCK_rel"] >= gate_corrupt0["mean"]["PCK_rel"] + 0.25
            and corrupt["mean"]["AUC_rel"] >= gate_corrupt0["mean"]["AUC_rel"] + 0.25
        )
        strict_source_gate = clean_preserved and corrupt_improved
        eligible = strict_source_gate if args.arm == "full" else clean_preserved
        score = sum(clean["mean"][key] + corrupt["mean"][key]
                    for key in ("PCK_rel", "AUC_rel"))
        if score > best_progress_score:
            best_progress_score, patience = score, 0
            torch.save({
                "state_dict": model.state_dict(),
                "metadata": {
                    "model": f"MuPoTSTargetedGraphMLP_{args.arm}",
                    "arm": args.arm,
                    "seed": args.seed,
                    "stage": args.stage,
                    "source_gate_pass": False,
                    "target_authorized": False,
                    "target_access": "none",
                    "purpose": "source-only optimization debugging",
                },
            }, progress_path)
        else:
            patience += 1
        if eligible and score > best_score:
            best_score, best_epoch = score, epoch
            torch.save({
                "state_dict": model.state_dict(),
                "metadata": {
                    "model": f"MuPoTSTargetedGraphMLP_{args.arm}",
                    "arm": args.arm,
                    "seed": args.seed,
                    "stage": args.stage,
                    "source_selected": True,
                    "target_access": "none",
                    "m2_checkpoint": str(Path(args.m2_checkpoint).resolve()),
                    "trainable_new_parameters": trainable,
                },
                "epoch0": baseline,
                "selected_clean": clean,
                "selected_corrupt": corrupt,
            }, best_path)
        row = {
            "epoch": epoch,
            **{key: float(np.mean(value)) for key, value in totals.items()},
            "clean_pck": clean["mean"]["PCK_rel"], "clean_auc": clean["mean"]["AUC_rel"],
            "corrupt_pck": corrupt["mean"]["PCK_rel"], "corrupt_auc": corrupt["mean"]["AUC_rel"],
            "eligible": eligible,
        }
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        print(json.dumps(row, sort_keys=True))
        if patience >= defaults["early_stop"]:
            break

    if best_epoch:
        status = "source_gate_pass" if args.arm == "full" else "source_selection_pass"
    else:
        status = "source_gate_fail" if args.arm == "full" else "source_selection_fail"
    summary = {
        "status": status,
        "arm": args.arm,
        "best_epoch": best_epoch,
        "best_score": best_score if best_epoch else None,
        "best_progress_score": best_progress_score,
        "device": str(device),
        "trainable_new_parameters": trainable,
        "target_access": "none",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
