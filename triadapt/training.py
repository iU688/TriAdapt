"""Source-domain optimization steps for feature fitting and component continuations."""
import torch
from .corruptions import apply_frozen_source_corruption
from .losses import (root_mpjpe_loss,paper_aligned_bone_loss,smooth_pck_loss,
                     finite_masked_smooth_l1)


def make_optimizer(model, lr=2e-4, weight_decay=1e-4):
    return torch.optim.AdamW(model.adaptation_parameters(),lr=lr,weight_decay=weight_decay)


def train_step(model, batch, optimizer, generator):
    """3D targets in mm; detector/gt2d in the same normalized screen coordinates."""
    model.train()
    device=next(model.parameters()).device
    tracks=batch['detector']
    # CPU generator is intentionally independent of CUDA RNG.
    if tracks.device.type != 'cpu': raise ValueError('Supply detector tensors on CPU')
    if model.stage != 'stage1':
        tracks=apply_frozen_source_corruption(tracks,generator,0.5)
    optimizer.zero_grad(set_to_none=True)
    pred,d=model(tracks.to(device),return_diagnostics=True)
    target=batch['target'].to(device)
    root=root_mpjpe_loss(pred,target)
    bone=paper_aligned_bone_loss(pred,target)
    if model.stage == 'stage1':
        loss=root+0.5*bone+0.01*d['adapter_gate'].abs().mean()
    else:
        repair=finite_masked_smooth_l1(d['corrected_2d'],batch['gt2d'].to(device),
                                      batch['gt_confidence'].to(device))
        loss=(root+0.5*bone+0.05*smooth_pck_loss(pred,target)
              +float(model.use_tcm)*repair
              +0.01*(d['correction_2d_rms']+d['refinement_3d_rms']))
    if not torch.isfinite(loss): raise ValueError('Nonfinite source loss')
    loss.backward()
    optimizer.step()
    return {'loss':float(loss.detach()),'root':float(root.detach()),'bone':float(bone.detach())}
