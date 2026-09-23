"""
Weighted Jacobi Convolution (WJ-Conv)
=======================================
Source: Spatio-Temporal MLP-Graph (BMVC 2023)
Reference file: _references/Spatio-Temporal-MLP-Graph/models/wj_conv.py

This file contains TWO classes:
1. WeightedJacobiConv — verbatim copy from ST-MLP-Graph reference
2. GcnWJ — wrapper adapting it to our (B, C, T, V) interface
"""
from __future__ import absolute_import, division

import math
import torch
import torch.nn as nn


# ──────────────────────────────────────────────
# 1. Original WeightedJacobiConv (verbatim)
# ──────────────────────────────────────────────
class WeightedJacobiConv(nn.Module):
    """
    Weighted jacobi graph convolution layer.
    Copied verbatim from ST-MLP-Graph reference code.
    """

    def __init__(self, in_features, out_features, adj, bias=True, ):
        super(WeightedJacobiConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W2 = nn.Parameter(torch.zeros(size=(1, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(1, adj.size(0), out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)

        self.adj_1 = adj
        self.adj2_1 = nn.Parameter(torch.ones_like(self.adj_1))
        nn.init.constant_(self.adj2_1, 1e-6)

        self.alpha = 0.1

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input1, input2):

        h1 = torch.matmul(input1, self.W[0])
        h2 = torch.matmul(input1, self.W[1])
        b = self.M[0] * h1

        adj = self.adj_1.to(input1.device) + self.adj2_1.to(input1.device)
        A_0 = (adj.T + adj) / 2
        c = torch.einsum('njc,jj->njc', (((1 - self.alpha) * self.M[0]) * h1), A_0)

        x1 = torch.matmul(input2, self.W2[0])
        e = self.alpha * self.M[0] * x1

        output = h2 - b + c + e

        if self.bias is not None:
            return output + self.bias.view(1, 1, -1), x1
        else:
            return output, x1

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# ──────────────────────────────────────────────
# 2. Wrapper for our framework: (B, C, T, V) interface
# ──────────────────────────────────────────────
class GcnWJ(nn.Module):
    """
    Wrapper that adapts WeightedJacobiConv to the (B, C, 1, V) interface
    used by our Block class in mlp_gcn.py.

    The original WJ-Conv operates on (N, J, C) format (batch, joints, channels).
    Our Block class calls gcn with (B, C, 1, V) and expects (B, C, 1, V) back.

    This wrapper handles the (B, C, 1, V) → (B, V, C) → WJ-Conv → (B, C, 1, V) conversion.
    """

    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        # adj from Graph class may be (hops, V, V); WJ-Conv needs (V, V) for self.M
        if adj.dim() == 3:
            adj_wj = adj[0]  # first hop: (V, V)
        else:
            adj_wj = adj
        self.register_buffer('adj', adj_wj)
        self.wj = WeightedJacobiConv(in_channels, out_channels, adj_wj)

    def forward(self, x):
        """
        Args:
            x: (B, C, 1, V) — standard Block gcn input format
              B=batch, C=channels, 1=temporal, V=vertex(joints)
        Returns:
            out: (B, C, 1, V)
        """
        # (B, C, 1, V) → squeeze → (B, C, V) → transpose → (B, V, C) ≡ (B, J, C)
        # WeightedJacobiConv使用 (N, J, C) 格式，其中 N=batch, J=joints, C=channels
        # self.M[0] shape 为 (J=17, C=out_features)，在 N 维广播
        x = x.squeeze(2).transpose(1, 2).contiguous()  # (B, V, C)

        # WJ-Conv operates on (N, J, C) format; feed same input for both streams
        out, _ = self.wj(x, x)  # (B, V, C)

        # (B, V, C) → transpose → (B, C, V) → unsqueeze → (B, C, 1, V)
        out = out.transpose(1, 2).unsqueeze(2).contiguous()  # (B, C, 1, V)
        return out
