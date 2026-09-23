"""R4-E GraphMLP variants with detector-independent temporal reliability."""

import math

import torch
import torch.nn as nn

from models.graphmlp import Model as SingleGraphMLP
from models.r2_pair_graphmlp import dct_lowpass_matrix


class TrackReliability(nn.Module):
    """Estimate joint reliability from visibility, jitter, and DCT residuals."""

    def __init__(self, frames, dct_keep, temperature=4.0):
        super().__init__()
        self.register_buffer("lowpass", dct_lowpass_matrix(frames, dct_keep))
        self.temperature = float(temperature)

    @staticmethod
    def robust_score(residual, valid, temperature):
        flat_residual = residual.flatten(2)
        flat_valid = valid.flatten(2)
        masked = flat_residual.masked_fill(~flat_valid, float("nan"))
        scale = torch.nanmedian(masked, dim=-1).values.nan_to_num(1e-4).clamp_min(1e-6)
        score = torch.exp(-residual / (temperature * scale[..., None, None]))
        return torch.where(valid, score, torch.ones_like(score))

    def forward(self, inputs):
        visible = torch.any(inputs != 0.0, dim=-1)
        smooth = torch.einsum("ts,bpsjc->bptjc", self.lowpass, inputs)
        dct_residual = torch.linalg.vector_norm(inputs - smooth, dim=-1)
        dct_score = self.robust_score(dct_residual, visible, self.temperature)

        acceleration = torch.zeros_like(inputs)
        acceleration[:, :, 1:-1] = (
            inputs[:, :, 1:-1]
            - 0.5 * (inputs[:, :, :-2] + inputs[:, :, 2:])
        )
        accel_valid = torch.zeros_like(visible)
        accel_valid[:, :, 1:-1] = (
            visible[:, :, :-2] & visible[:, :, 1:-1] & visible[:, :, 2:]
        )
        accel_residual = torch.linalg.vector_norm(acceleration, dim=-1)
        accel_score = self.robust_score(accel_residual, accel_valid, self.temperature)
        reliability = torch.sqrt(dct_score * accel_score) * visible.float()
        return reliability.detach(), smooth


class UnifiedTemporalRepair(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_strength = nn.Parameter(torch.zeros(()))

    @property
    def strength(self):
        return torch.tanh(self.raw_strength)

    def forward(self, inputs, reliability, smooth):
        candidate = reliability.unsqueeze(-1) * inputs + (1.0 - reliability.unsqueeze(-1)) * smooth
        return inputs + self.strength * (candidate - inputs)


class ReliabilityContextAdapter(nn.Module):
    def __init__(self, channel=512, bottleneck=32):
        super().__init__()
        self.norm = nn.LayerNorm(channel)
        self.query = nn.Linear(channel, bottleneck)
        self.key = nn.Linear(channel, bottleneck)
        self.value = nn.Linear(channel, bottleneck)
        self.geometry = nn.Sequential(nn.Linear(3, bottleneck), nn.GELU())
        self.output = nn.Linear(bottleneck, channel)
        self.gate = nn.Sequential(
            nn.Linear(6, bottleneck), nn.GELU(), nn.Linear(bottleneck, 1), nn.Sigmoid()
        )
        self.scale = math.sqrt(float(bottleneck))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def relative_geometry(anchors, mode):
        if mode == "self":
            return torch.zeros((*anchors.shape[:2], 3), dtype=anchors.dtype, device=anchors.device)
        source = anchors.flip(1)
        target_scale = anchors[..., 0].clamp_min(1e-6)
        source_scale = source[..., 0].clamp_min(1e-6)
        normalizer = torch.sqrt(target_scale * source_scale).clamp_min(1e-6)
        return torch.stack([
            torch.log(target_scale / source_scale),
            (source[..., 1] - anchors[..., 1]) / normalizer,
            (source[..., 2] - anchors[..., 2]) / normalizer,
        ], dim=-1).clamp(-10.0, 10.0)

    def forward(self, features, reliability, anchors, pair_is_distinct, mode,
                residual_scale=1.0):
        normalized = self.norm(features)
        source = normalized if mode == "self" else normalized.flip(1)
        geometry = self.relative_geometry(anchors, mode)
        query = self.query(normalized) + self.geometry(geometry).unsqueeze(2)
        key, value = self.key(source), self.value(source)
        attention = torch.softmax(
            torch.einsum("bpjd,bpkd->bpjk", query, key) / self.scale, dim=-1
        )
        context = torch.einsum("bpjk,bpkd->bpjd", attention, value)
        center = reliability[:, :, reliability.shape[2] // 2]
        target_rel = center.mean(-1).clamp(0.0, 1.0)
        source_rel = target_rel if mode == "self" else target_rel.flip(1)
        overlap = (center * (center if mode == "self" else center.flip(1))).mean(-1)
        uncertainty = 1.0 - target_rel
        gate_input = torch.cat([
            uncertainty.unsqueeze(-1), source_rel.unsqueeze(-1), overlap.unsqueeze(-1), geometry,
        ], dim=-1)
        gate = uncertainty * self.gate(gate_input).squeeze(-1)
        if mode == "cross":
            gate = gate * pair_is_distinct.float().unsqueeze(-1)
        return features + residual_scale * gate[..., None, None] * self.output(context), gate


class PairGeometryGate(nn.Module):
    """Bounded pair geometry modulation without companion pose features."""

    def __init__(self, bottleneck=32, max_scale=0.25):
        super().__init__()
        if not 0.0 < max_scale <= 1.0:
            raise ValueError("pair_gate_scale must be in (0, 1]")
        self.max_scale = float(max_scale)
        self.network = nn.Sequential(
            nn.Linear(6, bottleneck), nn.GELU(), nn.Linear(bottleneck, 1)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, reliability, anchors, pair_is_distinct):
        center = reliability[:, :, reliability.shape[2] // 2]
        target_rel = center.mean(-1).clamp(0.0, 1.0)
        source_rel = target_rel.flip(1)
        reliability_overlap = (center * center.flip(1)).mean(-1)
        geometry = ReliabilityContextAdapter.relative_geometry(anchors, "cross")
        pair_features = torch.cat([
            (1.0 - target_rel).unsqueeze(-1), source_rel.unsqueeze(-1),
            reliability_overlap.unsqueeze(-1), geometry,
        ], dim=-1)
        modulation = self.max_scale * torch.tanh(self.network(pair_features).squeeze(-1))
        return modulation * pair_is_distinct.float().unsqueeze(-1)


class R4EUnifiedGraphMLP(nn.Module):
    VARIANTS = {"repair", "self", "weak_cross", "hybrid", "pair_geometry"}

    def __init__(self, args, variant):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"unsupported R4-E variant: {variant}")
        self.variant = variant
        self.backbone = SingleGraphMLP(args)
        self.reliability = TrackReliability(args.frames, args.dct_keep)
        self.repair = UnifiedTemporalRepair()
        # Evaluation-only ablation switches. Training and normal inference keep
        # both branches enabled by default.
        self.disable_repair = False
        self.disable_self = False
        self.cross_residual_scale = float(getattr(args, "cross_residual_scale", 0.25))
        if not 0.0 < self.cross_residual_scale <= 1.0:
            raise ValueError("cross_residual_scale must be in (0, 1]")
        self.adapter = None
        self.self_adapter = None
        self.cross_adapter = None
        self.pair_gate = None
        if variant in {"self", "weak_cross"}:
            self.adapter = ReliabilityContextAdapter(args.channel, args.adapter_dim)
        elif variant == "hybrid":
            self.self_adapter = ReliabilityContextAdapter(args.channel, args.adapter_dim)
            self.cross_adapter = ReliabilityContextAdapter(args.channel, args.adapter_dim)
        elif variant == "pair_geometry":
            self.self_adapter = ReliabilityContextAdapter(args.channel, args.adapter_dim)
            self.pair_gate = PairGeometryGate(
                args.adapter_dim, float(getattr(args, "pair_gate_scale", 0.25))
            )

    def load_backbone_state(self, state_dict, strict=False):
        clean = {}
        for key, value in state_dict.items():
            key = key.replace("module.", "")
            if key.startswith("backbone."):
                key = key[len("backbone."):]
            parts = key.split(".")
            if len(parts) == 4 and parts[:2] == ["mlp_gcn", "blocks"] and parts[3] == "adj":
                continue
            if key.startswith(("reliability.", "repair.", "adapter.",
                               "self_adapter.", "cross_adapter.", "pair_gate.")):
                continue
            clean[key] = value
        return self.backbone.load_state_dict(clean, strict=strict)

    def load_hybrid_self_state(self, state_dict):
        """Initialize a frozen pair-aware branch from a trained R4-E self checkpoint."""
        if self.variant not in {"hybrid", "pair_geometry"}:
            raise ValueError("self warm-start requires hybrid or pair_geometry")
        clean = {key.replace("module.", ""): value for key, value in state_dict.items()}
        source_adapter = {
            key[len("adapter."):]: value for key, value in clean.items()
            if key.startswith("adapter.")
        }
        expected = set(self.self_adapter.state_dict())
        if set(source_adapter) != expected:
            missing = sorted(expected - set(source_adapter))
            unexpected = sorted(set(source_adapter) - expected)
            raise ValueError(
                f"self-adapter mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.self_adapter.load_state_dict(source_adapter, strict=True)
        repair_state = {
            key[len("repair."):]: value for key, value in clean.items()
            if key.startswith("repair.")
        }
        if set(repair_state) != set(self.repair.state_dict()):
            raise ValueError("self warm-start is missing the trained repair state")
        self.repair.load_state_dict(repair_state, strict=True)

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def freeze_hybrid_base(self):
        if self.variant not in {"hybrid", "pair_geometry"}:
            return
        for module in (self.repair, self.self_adapter):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def adaptation_parameters(self):
        return [parameter for name, parameter in self.named_parameters()
                if not name.startswith("backbone.") and parameter.requires_grad]

    def forward(self, inputs, anchors, pair_is_distinct, return_diagnostics=False):
        reliability, smooth = self.reliability(inputs)
        repaired = inputs if self.disable_repair else self.repair(inputs, reliability, smooth)
        batch, people, frames, joints, coords = repaired.shape
        flattened = repaired.reshape(batch * people, frames, joints, coords)
        _pose, features = self.backbone.forward_feat(flattened)
        features = features.reshape(batch, people, joints, -1)
        zero_gate = torch.zeros((batch, people), dtype=inputs.dtype, device=inputs.device)
        self_gate, cross_gate, pair_gate = zero_gate, zero_gate, zero_gate
        if self.variant in {"self", "weak_cross"}:
            mode = "self" if self.variant == "self" else "cross"
            if mode == "self" and self.disable_self:
                gate = zero_gate
            else:
                features, gate = self.adapter(
                    features, reliability, anchors, pair_is_distinct, mode
                )
            if mode == "self":
                self_gate = gate
            else:
                cross_gate = gate
        elif self.variant == "hybrid":
            features, self_gate = self.self_adapter(
                features, reliability, anchors, pair_is_distinct, "self"
            )
            features, cross_gate = self.cross_adapter(
                features, reliability, anchors, pair_is_distinct, "cross",
                residual_scale=self.cross_residual_scale,
            )
        elif self.variant == "pair_geometry":
            base_features = features
            self_features, self_gate = self.self_adapter(
                features, reliability, anchors, pair_is_distinct, "self"
            )
            pair_gate = self.pair_gate(reliability, anchors, pair_is_distinct)
            features = base_features + (
                1.0 + pair_gate[..., None, None]
            ) * (self_features - base_features)
        prediction = self.backbone.head(features)
        if return_diagnostics:
            if self.variant == "self":
                primary_gate = self_gate
            elif self.variant == "pair_geometry":
                primary_gate = pair_gate
            else:
                primary_gate = cross_gate
            return prediction, {
                "gate": primary_gate, "self_gate": self_gate,
                "cross_gate": cross_gate, "pair_gate": pair_gate,
                "reliability": reliability,
            }
        return prediction
