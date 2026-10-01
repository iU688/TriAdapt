"""FAM: shared joint attention with learned constant query offset and gate."""
import math
import torch
from torch import nn


class FeatureAdapter(nn.Module):
    def __init__(self, channel=512, bottleneck=32):
        super().__init__()
        self.norm = nn.LayerNorm(channel)
        self.query = nn.Linear(channel, bottleneck)
        self.key = nn.Linear(channel, bottleneck)
        self.value = nn.Linear(channel, bottleneck)
        # Keep the exact parameterization of the experimental implementation.
        # Both inputs are constant zero vectors, not person-pair geometry.
        self.geometry = nn.Sequential(nn.Linear(3, bottleneck), nn.GELU())
        self.output = nn.Linear(bottleneck, channel)
        self.gate = nn.Sequential(nn.Linear(6, bottleneck), nn.GELU(),
                                  nn.Linear(bottleneck, 1), nn.Sigmoid())
        self.scale = math.sqrt(float(bottleneck))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features):
        if features.ndim != 4:
            raise ValueError('features must have shape (B,P,J,C)')
        normalized = self.norm(features)
        zeros = features.new_zeros((*features.shape[:2], 3))
        query = self.query(normalized) + self.geometry(zeros).unsqueeze(2)
        key, value = self.key(normalized), self.value(normalized)
        attention = torch.softmax(torch.einsum('bpjd,bpkd->bpjk', query, key)
                                  / self.scale, dim=-1)
        context = torch.einsum('bpjk,bpkd->bpjd', attention, value)
        gate = self.gate(features.new_zeros((*features.shape[:2], 6))).squeeze(-1)
        return features + gate[..., None, None] * self.output(context), gate
