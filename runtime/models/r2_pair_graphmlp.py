"""R2 identity-preserving temporal repair and dual-person GraphMLP adapter."""

import math

import torch
import torch.nn as nn

from models.dual_joint_adapter import JointContextAdapter
from models.graphmlp import Model as SingleGraphMLP


def dct_lowpass_matrix(frames, keep):
    """Return the orthonormal DCT-II low-pass reconstruction matrix."""
    if not 1 <= keep <= frames:
        raise ValueError(f"dct_keep must be in [1,{frames}], got {keep}")
    time = torch.arange(frames, dtype=torch.float32)
    frequency = torch.arange(keep, dtype=torch.float32).unsqueeze(1)
    basis = torch.cos(math.pi / frames * (time + 0.5) * frequency)
    basis[0] *= math.sqrt(1.0 / frames)
    if keep > 1:
        basis[1:] *= math.sqrt(2.0 / frames)
    return basis.transpose(0, 1) @ basis


class ConfidenceTemporalRepair(nn.Module):
    """Blend low-frequency estimates only where confidence is weak.

    The scalar starts at zero, so adding this module cannot change the loaded
    GraphMLP prediction before training.
    """

    def __init__(self, frames, dct_keep, trainable=True):
        super().__init__()
        self.register_buffer("lowpass", dct_lowpass_matrix(frames, dct_keep))
        self.strength = nn.Parameter(torch.zeros(()), requires_grad=trainable)

    def forward(self, inputs, confidence):
        if confidence.shape != inputs.shape[:-1]:
            raise ValueError("confidence must match inputs without the xy dimension")
        confidence = confidence.clamp(0.0, 1.0).unsqueeze(-1)
        low_frequency = torch.einsum("ts,bpsjc->bptjc", self.lowpass, inputs)
        candidate = confidence * inputs + (1.0 - confidence) * low_frequency
        return inputs + self.strength * (candidate - inputs)


class R2PairGraphMLP(nn.Module):
    """Protected GraphMLP backbone plus optional repair and matched adapter."""

    VARIANTS = {"base", "repair", "self", "cross"}

    def __init__(self, args, variant):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unsupported R2 variant: {variant}")
        self.variant = variant
        self.backbone = SingleGraphMLP(args)
        self.repair = ConfidenceTemporalRepair(
            args.frames,
            getattr(args, "dct_keep", max(1, args.frames // 3)),
            trainable=variant != "base",
        )
        self.adapter = None
        if variant in {"self", "cross"}:
            self.adapter = JointContextAdapter(
                channel=args.channel,
                bottleneck=getattr(args, "adapter_dim", 32),
                gate_init=getattr(args, "adapter_gate_init", 0.1),
            )

    def load_backbone_state(self, state_dict, strict=False):
        clean = {}
        for key, value in state_dict.items():
            key = key.replace("module.", "")
            if key.startswith("backbone."):
                key = key[len("backbone."):]
            parts = key.split(".")
            if len(parts) == 4 and parts[:2] == ["mlp_gcn", "blocks"] and parts[2].isdigit() and parts[3] == "adj":
                continue
            if key.startswith(("adapter.", "repair.")):
                continue
            clean[key] = value
        return self.backbone.load_state_dict(clean, strict=strict)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def adaptation_parameters(self):
        return [parameter for name, parameter in self.named_parameters()
                if not name.startswith("backbone.") and parameter.requires_grad]

    def forward(self, inputs, confidence, anchors):
        if inputs.dim() != 5 or inputs.shape[1] != 2:
            raise ValueError(f"expected inputs (B,2,F,17,2), got {tuple(inputs.shape)}")
        repaired = self.repair(inputs, confidence)
        batch, people, frames, joints, coords = repaired.shape
        flattened = repaired.reshape(batch * people, frames, joints, coords)
        _pose, features = self.backbone.forward_feat(flattened)
        features = features.reshape(batch, people, joints, -1)
        if self.adapter is not None:
            features = self.adapter(features, anchors, self.variant)
        return self.backbone.head(features)

