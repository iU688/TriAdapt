"""Train the locked parameter-matched reliability-blind R4-F control.

This script deliberately mirrors ``train_r4e_equal_pilot.py``.  It reads only
the frozen source training/validation assets and refuses to write outside this
handoff.  MuPoTS and 3DPW-test paths are not accepted by the CLI.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

HANDOFF_ROOT = Path(__file__).resolve().parents[2]
if str(HANDOFF_ROOT) not in sys.path:
    sys.path.insert(0, str(HANDOFF_ROOT))

from common.pw3d_pair_dataset import PW3DPairDataset
from common.r3_multisource_dataset import MPIWeakPairDataset
from experiments.mechanism_controls.ungated_r4f import (
    R4FParameterMatchedUngatedGraphMLP,
)
import train_r4e_equal_pilot as formal_training
from train_r4e_equal_pilot import (
    evaluate,
    load_backbone,
    mean_metrics,
    set_seed,
    train_epoch,
)

# Keep long-run logs bounded so a real exception is never displaced by tens of
# thousands of terminal progress updates. This changes display only; loaders,
# ordering, RNG, gradients, and selection are untouched.
formal_training.tqdm = lambda iterable, **_kwargs: iterable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Locked parameter-matched reliability-blind R4-F control"
    )
    parser.add_argument("--mpi_train", required=True)
    parser.add_argument("--mpi_val", required=True)
    parser.add_argument("--pw3d_train", required=True)
    parser.add_argument("--pw3d_val", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--frames", type=int, default=243)
    parser.add_argument("--dct_keep", type=int, default=27)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--early_stop", type=int, default=5)
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
    parser.add_argument("--seed", type=int, choices=(42, 123, 2026), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--verify_epoch1_csv",
        help="infrastructure-retry guard: require exact epoch-1 equality before epoch 2",
    )
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


def handoff_output(path_text):
    path = Path(path_text).resolve()
    try:
        path.relative_to(HANDOFF_ROOT)
    except ValueError as error:
        raise ValueError(f"output_dir must be inside {HANDOFF_ROOT}") from error
    path.mkdir(parents=True, exist_ok=True)
    return path


def main():
    args = parse_args()
    args.variant = R4FParameterMatchedUngatedGraphMLP.CONTROL_VARIANT
    if args.frames != 243 or args.dct_keep != 27:
        raise ValueError("formal mechanism control is locked to frames=243,dct_keep=27")
    if args.num_workers != 0:
        raise ValueError("num_workers is locked to 0 to avoid external worker caches")
    ratios = np.asarray([args.mpi_ratio, args.pw3d_ratio], dtype=np.float64)
    if np.any(ratios <= 0.0) or not np.isclose(ratios.sum(), 1.0):
        raise ValueError("source ratios must be positive and sum to 1")
    out_dir = handoff_output(args.output_dir)
    checkpoint_dir, log_dir = out_dir / "checkpoints", out_dir / "logs"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mpi_train = MPIWeakPairDataset(
        args.mpi_train, args.frames, augment_flip=True, seed=args.seed
    )
    pw3d_train = PW3DPairDataset(
        args.pw3d_train,
        args.frames,
        center_stride=1,
        augment_flip=True,
        seed=args.seed,
        include_single=False,
    )
    mpi_val = MPIWeakPairDataset(
        args.mpi_val, args.frames, augment_flip=False, seed=args.seed
    )
    pw3d_val = PW3DPairDataset(args.pw3d_val, args.frames, center_stride=1)
    train_sets = [mpi_train, pw3d_train]
    combined = ConcatDataset(train_sets)
    weights = torch.cat(
        [
            torch.full(
                (len(dataset),), float(ratio / len(dataset)), dtype=torch.double
            )
            for dataset, ratio in zip(train_sets, ratios)
        ]
    )
    sampler = WeightedRandomSampler(
        weights,
        args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        combined,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=True,
    )
    mpi_loader = DataLoader(mpi_val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pw3d_loader = DataLoader(pw3d_val, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = R4FParameterMatchedUngatedGraphMLP(args).to(device)
    load_backbone(model, args.checkpoint, device)
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        model.adaptation_parameters(), lr=args.lr, weight_decay=1e-4
    )
    trainable = sum(parameter.numel() for parameter in model.adaptation_parameters())
    print("[Protocol] exact R4-F source data/sampling/selection; target access=none")
    print(f"[Control] reliability-blind constant gate; trainable={trainable}")
    print(f"[Output] {out_dir}")

    mpi0, pw3d0 = evaluate(model, mpi_loader, device), evaluate(model, pw3d_loader, device)
    mean0 = mean_metrics(mpi0, pw3d0)
    print(
        f"[Epoch 0] mean={mean0['PCK_rel']:.2f}/{mean0['AUC_rel']:.2f} "
        f"MPI={mpi0['PCK_rel']:.2f}/{mpi0['AUC_rel']:.2f} "
        f"3DPW={pw3d0['PCK_rel']:.2f}/{pw3d0['AUC_rel']:.2f}"
    )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{args.tag}_{stamp}.csv"
    best_path = checkpoint_dir / f"best_{args.tag}.pth"
    fields = [
        "epoch", "loss", "root_loss", "paper_loss", "gate", "self_gate",
        "cross_gate", "pair_gate", "pair_gate_abs", "repair_strength",
        "mpi_samples", "pw3d_samples", "distinct_samples", "mpi_pck", "mpi_auc",
        "pw3d_pck", "pw3d_auc", "mean_pck", "mean_auc", "eligible",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best_score = mean0["PCK_rel"] + mean0["AUC_rel"]
    best_epoch, patience = 0, 0
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
        eligible = (
            domains_preserved
            and mean["PCK_rel"] >= mean0["PCK_rel"] + args.min_mean_gain
            and mean["AUC_rel"] >= mean0["AUC_rel"] + args.min_mean_gain
        )
        score = mean["PCK_rel"] + mean["AUC_rel"]
        if eligible and score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mechanism_control_metadata": {
                        **vars(args),
                        "variant": R4FParameterMatchedUngatedGraphMLP.CONTROL_VARIANT,
                        "target_access_during_training_selection": "none",
                    },
                    "epoch0_mpi": mpi0,
                    "epoch0_pw3d": pw3d0,
                },
                best_path,
            )
            print(f"  [best eligible] saved {best_path}")
        else:
            patience += 1
        row = {
            "epoch": epoch, "loss": train["loss"], "root_loss": train["root_loss"],
            "paper_loss": train["paper_loss"], "gate": train["gate"],
            "self_gate": train["self_gate"], "cross_gate": train["cross_gate"],
            "pair_gate": train["pair_gate"], "pair_gate_abs": train["pair_gate_abs"],
            "repair_strength": float(model.repair.strength.item()),
            "mpi_samples": train["mpi"], "pw3d_samples": train["pw3d"],
            "distinct_samples": train["distinct"], "mpi_pck": mpi["PCK_rel"],
            "mpi_auc": mpi["AUC_rel"], "pw3d_pck": pw3d["PCK_rel"],
            "pw3d_auc": pw3d["AUC_rel"], "mean_pck": mean["PCK_rel"],
            "mean_auc": mean["AUC_rel"], "eligible": eligible,
        }
        with log_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        if epoch == 1 and args.verify_epoch1_csv:
            with Path(args.verify_epoch1_csv).open(newline="", encoding="utf-8") as handle:
                reference_rows = list(csv.DictReader(handle))
            if not reference_rows:
                raise ValueError("epoch-1 reference CSV contains no data row")
            expected = reference_rows[0]
            actual = {field: str(row[field]) for field in fields}
            mismatches = {
                field: {"expected": expected[field], "actual": actual[field]}
                for field in fields if expected[field] != actual[field]
            }
            if mismatches:
                raise RuntimeError(f"retry epoch-1 mismatch: {mismatches}")
            print(f"[Retry guard] epoch-1 exact field match: {args.verify_epoch1_csv}")
        print(
            f"[Epoch {epoch}] mean={mean['PCK_rel']:.2f}/{mean['AUC_rel']:.2f} "
            f"MPI={mpi['PCK_rel']:.2f}/{mpi['AUC_rel']:.2f} "
            f"3DPW={pw3d['PCK_rel']:.2f}/{pw3d['AUC_rel']:.2f} "
            f"gate={train['gate']:.4f} eligible={eligible}"
        )
        if args.early_stop > 0 and patience >= args.early_stop:
            print(f"[Early stop] no eligible source improvement for {patience} epochs")
            break
    print(f"[Done] best_epoch={best_epoch} best_score={best_score:.4f}")
    print(f"[Log] {log_path}")
    if best_epoch == 0:
        print("[Selection gate] FAIL: no target evaluation is permitted")


if __name__ == "__main__":
    main()
