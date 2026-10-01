"""Adapter integration using an externally constructed and loaded pose lifter."""
import torch
from torch import nn
from .fam import FeatureAdapter
from .geometry import TemporalGraph2DCorrector, KinematicGraphRefiner


class TriAdapt(nn.Module):
    ARMS = ('fam', 'fam_tcm', 'fam_kgr', 'triadapt')

    def __init__(self, backbone, channel=512, fam_width=32, tcm_width=16,
                 kgr_width=64, arm='triadapt', stage='joint'):
        super().__init__()
        if not hasattr(backbone, 'head'):
            raise ValueError('External backbone must expose its pre-head joint features at .head')
        self.backbone = backbone
        self.tcm = TemporalGraph2DCorrector(hidden=tcm_width)
        self.fam = FeatureAdapter(channel, fam_width)
        self.kgr = KinematicGraphRefiner(hidden=kgr_width)
        self.configure(arm, stage)

    def configure(self, arm, stage):
        if arm not in self.ARMS or stage not in ('stage1', 'stage2', 'joint'):
            raise ValueError('Invalid arm or training stage')
        if stage == 'stage1' and arm != 'fam':
            raise ValueError('Stage 1 trains FAM only')
        if stage == 'stage2' and arm == 'fam':
            raise ValueError('Stage 2 requires TCM and/or KGR')
        self.arm, self.stage = arm, stage
        self.use_tcm = arm in ('fam_tcm','triadapt')
        self.use_kgr = arm in ('fam_kgr','triadapt')
        self.backbone.requires_grad_(False)
        self.fam.requires_grad_(stage != 'stage2')
        self.tcm.requires_grad_(self.use_tcm)
        self.kgr.requires_grad_(self.use_kgr)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        if getattr(self,'stage',None) == 'stage2': self.fam.eval()
        return self

    def adaptation_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def adapter_state(self):
        return {name: module.state_dict() for name,module in
                [('tcm',self.tcm),('fam',self.fam),('kgr',self.kgr)]}

    def load_adapter_state(self, state):
        for name in ('tcm','fam','kgr'):
            getattr(self,name).load_state_dict(state[name], strict=True)

    def forward(self, tracks, return_diagnostics=False):
        if tracks.ndim != 5 or tracks.shape[-2:] != (17,2):
            raise ValueError('tracks must have shape (B,P,T,17,2)')
        b,p,t,j,c = tracks.shape
        corrected,delta = self.tcm(tracks) if self.use_tcm else (tracks,torch.zeros_like(tracks))
        diagnostic = {}
        # A temporary pre-head hook adds FAM without distributing backbone code.
        # Use separate model instances for concurrent threaded inference.
        def adapt_head(_module, args):
            features = args[0]
            if features.shape[:2] != (b*p,j) or features.ndim != 3:
                raise ValueError('Expected pre-head features (B*P,17,C)')
            adapted,gate = self.fam(features.reshape(b,p,j,-1))
            diagnostic['adapter_gate'] = gate
            return (adapted.reshape_as(features), *args[1:])
        hook = self.backbone.head.register_forward_pre_hook(adapt_head)
        try:
            prediction = self.backbone(corrected.reshape(b*p,t,j,c))
        finally:
            hook.remove()
        if 'adapter_gate' not in diagnostic:
            raise ValueError('External backbone did not call .head')
        if prediction.shape not in ((b*p,1,j,3),(b*p,j,3)):
            raise ValueError('External backbone output must be (B*P,1,17,3) or (B*P,17,3)')
        prediction = prediction.reshape(b,p,j,3)
        if self.use_kgr:
            prediction,refinement = self.kgr(prediction)
        else:
            prediction = prediction - prediction[:,:,0:1]
            refinement = torch.zeros_like(prediction)
        diagnostic.update(corrected_2d=corrected,correction_2d=delta,refinement_3d=refinement,
            correction_2d_rms=(delta.square().mean()+1e-12).sqrt(),
            refinement_3d_rms=(refinement.square().mean()+1e-12).sqrt())
        return (prediction,diagnostic) if return_diagnostics else prediction
