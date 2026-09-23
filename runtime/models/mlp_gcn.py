from functools import partial
from einops import rearrange
import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

from models.wj_conv import GcnWJ


def _make_gcn(in_channels, out_channels, adj, gcn_type='semgraph'):
    """Factory: create GCN layer by type.

    Args:
        gcn_type: 'semgraph' (default, original GraphMLP), 'wj' (WeightedJacobiConv)
    """
    if gcn_type == 'wj':
        return GcnWJ(in_channels, out_channels, adj)
    elif gcn_type == 'semgraph':
        return Gcn(in_channels, out_channels, adj)
    else:
        raise ValueError(f"Unknown gcn_type: {gcn_type}. Use 'semgraph' or 'wj'.")


class Gcn(nn.Module):
    def __init__(self, in_channels, out_channels, adj):
        super().__init__()
        self.register_buffer('adj', adj)
        self.kernel_size = adj.size(0)
        self.conv = nn.Conv2d(in_channels, out_channels * self.kernel_size, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x_out = torch.einsum('nkctv, kvw -> nctw', x, self.adj)
        return x_out.contiguous()

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x

class Mlp_ln(nn.Module):
    """Mlp with LayerNorm inside each fc (original GraphMLP, used when frames>1)"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features)
        )
        self.act = act_layer()
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_features, out_features),
            nn.LayerNorm(out_features)
        )
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, length, frames, dim, tokens_dim, channels_dim, adj, drop=0.1, drop_path=0., gcn_type='semgraph'):
        super().__init__()
        self.norm1 = nn.LayerNorm(length)
        self.gcn_1 = _make_gcn(dim, dim, adj, gcn_type=gcn_type)
        if frames == 1:
            self.mlp_1 = Mlp(in_features=length, hidden_features=tokens_dim, drop=drop)
        else:
            self.mlp_1 = Mlp_ln(in_features=length, hidden_features=tokens_dim, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.gcn_2 = _make_gcn(dim, dim, adj, gcn_type=gcn_type)
        self.mlp_2 = Mlp(in_features=dim, hidden_features=channels_dim, drop=drop)

    def forward(self, x):
        x_s = rearrange(x, 'b j c -> b c j')
        res_s = x_s
        x_s = self.norm1(x_s) 
        
        x_gcn_1 = rearrange(x_s, 'b c j -> b c 1 j') 
        x_gcn_1 = self.gcn_1(x_gcn_1)
        x_gcn_1 = rearrange(x_gcn_1, 'b c 1 j -> b c j') 
        x_s_out = res_s + self.drop_path(self.mlp_1(x_s) + x_gcn_1) 
        
        x_c = rearrange(x_s_out, 'b c j -> b j c')
        res_c = x_c
        x_c = self.norm2(x_c) 
        
        x_gcn_2 = rearrange(x_c, 'b j c -> b c 1 j') 
        x_gcn_2 = self.gcn_2(x_gcn_2)
        x_gcn_2 = rearrange(x_gcn_2, 'b c 1 j -> b j c') 
        x_out = res_c + self.drop_path(self.mlp_2(x_c) + x_gcn_2) 
        
        return x_out

class Mlp_gcn(nn.Module):
    def __init__(self, depth, embed_dim, channels_dim, tokens_dim, adj, drop_rate=0.1, length=17, frames=1, gcn_type='semgraph'):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, 0.2, depth)]
        self.blocks = nn.ModuleList([Block(length, frames, embed_dim, tokens_dim, channels_dim, adj,
                                           drop=drop_rate, drop_path=dpr[i], gcn_type=gcn_type) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)