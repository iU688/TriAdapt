import torch
import torch.nn.functional as F
from .geometry import BONES
EVAL_JOINTS_14 = [1,2,3,4,5,6,8,10,11,12,13,14,15,16]

def root_relative(prediction, target_mm):
    target = target_mm / 1000.0
    prediction = prediction - prediction[:, :, 0:1]
    target = target - target[:, :, 0:1]
    return prediction, target


def root_mpjpe_loss(prediction, target_mm):
    prediction, target = root_relative(prediction, target_mm)
    return torch.linalg.vector_norm(prediction - target, dim=-1).mean()


def paper_aligned_bone_loss(prediction, target_mm):
    """Differentiable counterpart of the reported bone-normalized 14-joint error."""
    prediction, target = root_relative(prediction, target_mm)
    pred_lengths, target_lengths = [], []
    for parent, child in BONES:
        pred_lengths.append(torch.linalg.vector_norm(
            prediction[:, :, child] - prediction[:, :, parent], dim=-1
        ))
        target_lengths.append(torch.linalg.vector_norm(
            target[:, :, child] - target[:, :, parent], dim=-1
        ))
    pred_scale = torch.stack(pred_lengths, dim=-1).mean(-1).clamp_min(1e-6)
    target_scale = torch.stack(target_lengths, dim=-1).mean(-1).clamp_min(1e-6)
    normalized = prediction * (target_scale / pred_scale)[..., None, None]
    joints = torch.as_tensor(EVAL_JOINTS_14, dtype=torch.long, device=prediction.device)
    return torch.linalg.vector_norm(
        normalized.index_select(2, joints) - target.index_select(2, joints), dim=-1
    ).mean()

def finite_masked_smooth_l1(prediction, target, confidence):
    """Smooth-L1 that never evaluates arithmetic on non-finite target entries."""
    finite = torch.isfinite(target).all(dim=-1, keepdim=True)
    mask = (confidence.unsqueeze(-1) > 0.0) & finite
    safe_prediction = torch.where(mask, prediction, torch.zeros_like(prediction))
    safe_target = torch.where(mask, target, torch.zeros_like(target))
    per_coordinate = F.smooth_l1_loss(safe_prediction, safe_target, reduction="none")
    return per_coordinate.sum() / (mask.sum().clamp_min(1) * prediction.shape[-1])


def smooth_pck_loss(prediction, target_mm, threshold_m=0.15, temperature_m=0.02):
    prediction, target = root_relative(prediction, target_mm)
    pred_lengths, target_lengths = [], []
    for parent, child in BONES:
        pred_lengths.append(torch.linalg.vector_norm(
            prediction[:, :, child] - prediction[:, :, parent], dim=-1
        ))
        target_lengths.append(torch.linalg.vector_norm(
            target[:, :, child] - target[:, :, parent], dim=-1
        ))
    pred_scale = torch.stack(pred_lengths, dim=-1).mean(-1).clamp_min(1e-6)
    target_scale = torch.stack(target_lengths, dim=-1).mean(-1).clamp_min(1e-6)
    aligned = prediction * (target_scale / pred_scale)[..., None, None]
    joints = torch.as_tensor(EVAL_JOINTS_14, device=prediction.device)
    error = torch.linalg.vector_norm(
        aligned.index_select(2, joints) - target.index_select(2, joints), dim=-1
    )
    return torch.sigmoid((error - threshold_m) / temperature_m).mean()
