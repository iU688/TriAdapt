"""Strict loader for source-selected reliability-blind control checkpoints."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from experiments.mechanism_controls.ungated_r4f import (
    R4FParameterMatchedUngatedGraphMLP,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_control_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "mechanism_control_metadata" not in checkpoint:
        raise ValueError("expected a source-selected mechanism-control checkpoint")
    metadata = dict(checkpoint["mechanism_control_metadata"])
    if metadata.get("variant") != R4FParameterMatchedUngatedGraphMLP.CONTROL_VARIANT:
        raise ValueError("unexpected mechanism-control variant")
    if metadata.get("target_access_during_training_selection") != "none":
        raise ValueError("checkpoint does not prove target-free training/selection")
    model_args = SimpleNamespace(
        frames=int(metadata["frames"]), dct_keep=int(metadata["dct_keep"]),
        channel=int(metadata.get("channel", 512)), d_hid=int(metadata.get("d_hid", 1024)),
        token_dim=int(metadata.get("token_dim", 256)), layers=int(metadata.get("layers", 3)),
        n_joints=int(metadata.get("n_joints", 17)),
        drop_rate=float(metadata.get("drop_rate", 0.1)),
        adapter_dim=int(metadata.get("adapter_dim", 32)),
        cross_residual_scale=float(metadata.get("cross_residual_scale", 0.25)),
        pair_gate_scale=float(metadata.get("pair_gate_scale", 0.25)),
    )
    model = R4FParameterMatchedUngatedGraphMLP(model_args).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, metadata

