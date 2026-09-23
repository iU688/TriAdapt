"""Lightweight permutation-equivariant joint-level dual-person adapter."""

import math

import torch
import torch.nn as nn

from models.graphmlp import Model as SingleGraphMLP


class JointContextAdapter(nn.Module):
    """Apply shared joint attention using either self or the other person."""

    def __init__(self, channel=512, bottleneck=32, gate_init=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channel)
        self.query = nn.Linear(channel, bottleneck)
        self.key = nn.Linear(channel, bottleneck)
        self.value = nn.Linear(channel, bottleneck)
        self.geometry = nn.Sequential(
            nn.Linear(3, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, bottleneck),
        )
        self.output = nn.Linear(bottleneck, channel)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self.scale = math.sqrt(float(bottleneck))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _relative_geometry(anchor, source_anchor):
        target_scale = anchor[..., 0].clamp_min(1e-6)
        source_scale = source_anchor[..., 0].clamp_min(1e-6)
        normalizer = torch.sqrt(target_scale * source_scale).clamp_min(1e-6)
        geometry = torch.stack([
            torch.log(target_scale / source_scale),
            (source_anchor[..., 1] - anchor[..., 1]) / normalizer,
            (source_anchor[..., 2] - anchor[..., 2]) / normalizer,
        ], dim=-1)
        return geometry.clamp(-10.0, 10.0)

    def forward(self, features, anchors, context_mode):
        if context_mode not in {'self', 'cross'}:
            raise ValueError(f'context_mode must be self or cross, got {context_mode}')
        normalized = self.norm(features)
        if context_mode == 'cross':
            source = normalized.flip(1)
            source_anchor = anchors.flip(1)
            geometry = self.geometry(self._relative_geometry(anchors, source_anchor))
        else:
            source = normalized
            geometry_input = torch.zeros(
                (*features.shape[:2], 3),
                dtype=features.dtype,
                device=features.device,
            )
            geometry = self.geometry(geometry_input)
        query = self.query(normalized) + geometry.unsqueeze(2)
        key = self.key(source)
        value = self.value(source)
        attention = torch.softmax(
            torch.einsum('bpjd,bpkd->bpjk', query, key) / self.scale,
            dim=-1,
        )
        context = torch.einsum('bpjk,bpkd->bpjd', attention, value)
        return features + self.gate * self.output(context)


class DualJointAdapterGraphMLP(nn.Module):
    """Shared GraphMLP plus a small joint-level self/cross context adapter."""

    def __init__(self, args, context_mode='cross'):
        super().__init__()
        if context_mode not in {'none', 'self', 'cross'}:
            raise ValueError(f'unsupported context_mode={context_mode}')
        self.context_mode = context_mode
        self.backbone = SingleGraphMLP(args)
        self.adapter = None
        if context_mode != 'none':
            self.adapter = JointContextAdapter(
                channel=args.channel,
                bottleneck=getattr(args, 'adapter_dim', 32),
                gate_init=getattr(args, 'adapter_gate_init', 0.1),
            )

    def load_backbone_state(self, state_dict, strict=False):
        clean = {}
        for key, value in state_dict.items():
            normalized = key.replace('module.', '')
            if normalized.startswith('backbone.'):
                normalized = normalized[len('backbone.'):]
            parts = normalized.split('.')
            if (len(parts) == 4 and parts[0] == 'mlp_gcn' and
                    parts[1] == 'blocks' and parts[2].isdigit() and parts[3] == 'adj'):
                continue
            if normalized.startswith('adapter.'):
                continue
            clean[normalized] = value
        return self.backbone.load_state_dict(clean, strict=strict)

    def forward(self, inputs, anchors):
        if inputs.dim() != 5 or inputs.shape[1] != 2:
            raise ValueError(f'expected (B,2,F,17,2), got {tuple(inputs.shape)}')
        batch, people, frames, joints, coords = inputs.shape
        flattened = inputs.reshape(batch * people, frames, joints, coords)
        _pose, features = self.backbone.forward_feat(flattened)
        features = features.reshape(batch, people, joints, -1)
        if self.context_mode != 'none':
            features = self.adapter(features, anchors, self.context_mode)
        return self.backbone.head(features)
