import torch
import torch.nn as nn
from einops import rearrange
from models.graph_frames import Graph
from models.mlp_gcn import Mlp_gcn


class Model(nn.Module):
    """Clean single-person GraphMLP, faithful to the original paper.

    Input:  (B, F, J, 2)   J = 17 (H36M)
    Output: (B, 1, J, 3)
    Multi-person is handled OUTSIDE this module (batch-folding wrapper), so the
    network itself stays exactly the single-person GraphMLP.

    Original GraphMLP innovations:
      - embedding → L × Block(Spatial Graph MLP + Channel Graph MLP) → head
      - Multi-adjacency GCN (K=4: self/close/further/sym)
      - DropPath linear schedule [0→0.2]
    """
    def __init__(self, args):
        super().__init__()
        self.graph = Graph('hm36_gt', 'spatial', pad=1)
        self.A = nn.Parameter(torch.tensor(self.graph.A, dtype=torch.float32), requires_grad=False)  # (K,17,17)

        # Original GraphMLP embedding: 2*F → channel
        self.embedding = nn.Linear(2 * args.frames, args.channel)
        drop_rate = getattr(args, 'drop_rate', 0.1)
        self.mlp_gcn = Mlp_gcn(args.layers, args.channel, args.d_hid, args.token_dim, self.A,
                               drop_rate=drop_rate, length=args.n_joints, frames=args.frames)
        self.head = nn.Linear(args.channel, 3)

    def forward(self, x):
        x = rearrange(x, 'b f j c -> b j (c f)').contiguous()  # B J (2F)
        x = self.embedding(x)       # B J C
        x = self.mlp_gcn(x)         # B J C
        x = self.head(x)            # B J 3
        x = rearrange(x, 'b j c -> b 1 j c').contiguous()      # B 1 J 3
        return x

    def forward_feat(self, x):
        """Same as forward but ALSO returns the pre-head joint features.

        Returns:
            out:  (B, 1, J, 3) 3D pose (identical to forward())
            feat: (B, J, C)    pre-head features (for cross-person interaction)
        """
        x = rearrange(x, 'b f j c -> b j (c f)').contiguous()  # B J (2F)
        x = self.embedding(x)       # B J C
        feat = self.mlp_gcn(x)      # B J C
        out = self.head(feat)       # B J 3
        out = rearrange(out, 'b j c -> b 1 j c').contiguous()  # B 1 J 3
        return out, feat