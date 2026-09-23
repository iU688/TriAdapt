from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / "runtime"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from experiments.mechanism_controls.control_checkpoint import load_control_checkpoint


BONES = (
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16),
)


def normalized_adjacency(joints: int = 17) -> torch.Tensor:
    matrix = torch.eye(joints, dtype=torch.float32)
    for parent, child in BONES:
        matrix[parent, child] = 1.0
        matrix[child, parent] = 1.0
    degree = matrix.sum(-1).clamp_min(1.0)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * matrix * inv_sqrt[None, :]


def track_scale(inputs: torch.Tensor) -> torch.Tensor:
    lengths = []
    for parent, child in BONES:
        lengths.append(torch.linalg.vector_norm(
            inputs[..., child, :] - inputs[..., parent, :], dim=-1
        ))
    mean_length = torch.stack(lengths, dim=-1).mean(-1)
    return mean_length.median(dim=-1).values.clamp(0.02, 0.5)


class TemporalGraph2DCorrector(nn.Module):
    """Small residual corrector for dense-but-corrupted 2D tracks."""

    def __init__(self, hidden: int = 16, max_relative_correction: float = 0.35):
        super().__init__()
        self.hidden = int(hidden)
        self.max_relative_correction = float(max_relative_correction)
        self.input_projection = nn.Linear(6, hidden)
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2, groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=4, dilation=2, groups=hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 1),
        )
        self.graph_norm = nn.LayerNorm(hidden)
        self.graph_projection = nn.Linear(hidden, hidden)
        self.output_projection = nn.Linear(hidden, 2)
        self.register_buffer("adjacency", normalized_adjacency())
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 5 or inputs.shape[-2:] != (17, 2):
            raise ValueError("inputs must have shape (B,P,T,17,2)")
        batch, people, frames, joints, _ = inputs.shape
        flat = inputs.reshape(batch * people, frames, joints, 2)
        smooth = F.avg_pool1d(
            flat.permute(0, 2, 3, 1).reshape(-1, 2, frames),
            kernel_size=5,
            stride=1,
            padding=2,
        ).reshape(batch * people, joints, 2, frames).permute(0, 3, 1, 2)
        acceleration = torch.zeros_like(flat)
        acceleration[:, 1:-1] = flat[:, 1:-1] - 0.5 * (flat[:, :-2] + flat[:, 2:])
        features = torch.cat((flat, flat - smooth, acceleration), dim=-1)
        hidden = self.input_projection(features)
        temporal = self.temporal(
            hidden.permute(0, 2, 3, 1).reshape(-1, self.hidden, frames)
        ).reshape(batch * people, joints, self.hidden, frames).permute(0, 3, 1, 2)
        graph = torch.einsum("jk,btkh->btjh", self.adjacency, hidden)
        fused = temporal + self.graph_projection(self.graph_norm(graph))
        raw_delta = self.output_projection(F.gelu(fused))
        scale = track_scale(flat)[:, None, None, None]
        delta = self.max_relative_correction * scale * torch.tanh(raw_delta)
        corrected = flat + delta
        return corrected.reshape_as(inputs), delta.reshape_as(inputs)


class KinematicGraphBlock(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.self_projection = nn.Linear(hidden, hidden)
        self.graph_projection = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, hidden: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        graph = torch.einsum("jk,bpkh->bpjh", adjacency, hidden)
        update = F.gelu(self.norm(
            self.self_projection(hidden) + self.graph_projection(graph)
        ))
        return hidden + update


class KinematicGraphRefiner(nn.Module):
    """Zero-initialized pose-level graph residual with pelvis recentering."""

    def __init__(self, hidden: int = 64, blocks: int = 2):
        super().__init__()
        self.input_projection = nn.Linear(3, hidden)
        self.blocks = nn.ModuleList([KinematicGraphBlock(hidden) for _ in range(blocks)])
        self.output_projection = nn.Linear(hidden, 3)
        self.register_buffer("adjacency", normalized_adjacency())
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pose.ndim != 4 or pose.shape[-2:] != (17, 3):
            raise ValueError("pose must have shape (B,P,17,3)")
        centered = pose - pose[:, :, 0:1]
        hidden = self.input_projection(centered)
        for block in self.blocks:
            hidden = block(hidden, self.adjacency)
        delta = self.output_projection(hidden)
        refined = centered + delta
        refined = refined - refined[:, :, 0:1]
        return refined, refined - centered


class MuPoTSTargetedGraphMLP(nn.Module):
    """M1 -> frozen GraphMLP+M2 -> M3, with no companion information."""

    def __init__(self, m2_checkpoint: str | Path, device: torch.device | str = "cpu"):
        super().__init__()
        device = torch.device(device)
        self.base, self.base_metadata = load_control_checkpoint(m2_checkpoint, device)
        self.base.disable_repair = True
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.corrector = TemporalGraph2DCorrector()
        self.refiner = KinematicGraphRefiner()
        self.disable_corrector = False
        self.disable_refiner = False
        self.active_arm = "full"
        self.configure_arm("full")

    def configure_arm(self, arm: str) -> None:
        """Activate a frozen formal arm without changing checkpoint structure."""
        if arm not in {"m1_m2", "m2_m3", "full"}:
            raise ValueError(f"unsupported arm: {arm}")
        use_corrector = arm in {"m1_m2", "full"}
        use_refiner = arm in {"m2_m3", "full"}
        self.disable_corrector = not use_corrector
        self.disable_refiner = not use_refiner
        for parameter in self.corrector.parameters():
            parameter.requires_grad = use_corrector
        for parameter in self.refiner.parameters():
            parameter.requires_grad = use_refiner
        self.active_arm = arm

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.eval()
        return self

    def adaptation_parameters(self):
        return [parameter for name, parameter in self.named_parameters()
                if not name.startswith("base.") and parameter.requires_grad]

    def forward(self, inputs, anchors, pair_is_distinct, return_diagnostics=False):
        if self.disable_corrector:
            corrected, correction = inputs, torch.zeros_like(inputs)
        else:
            corrected, correction = self.corrector(inputs)
        prediction, base_diagnostics = self.base(
            corrected, anchors, pair_is_distinct, return_diagnostics=True
        )
        if self.disable_refiner:
            refined = prediction - prediction[:, :, 0:1]
            refinement = torch.zeros_like(refined)
        else:
            refined, refinement = self.refiner(prediction)
        if return_diagnostics:
            diagnostics = dict(base_diagnostics)
            diagnostics.update({
                "corrected_2d": corrected,
                "correction_2d": correction,
                "refinement_3d": refinement,
                "correction_2d_rms": (correction.square().mean() + 1e-12).sqrt(),
                "refinement_3d_rms": (refinement.square().mean() + 1e-12).sqrt(),
            })
            return refined, diagnostics
        return refined
