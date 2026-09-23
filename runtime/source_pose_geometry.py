"""Skeleton geometry for source validation."""
import numpy as np

H36M_BONES = [
    (0, 1), (1, 2), (2, 3),        # R leg
    (0, 4), (4, 5), (5, 6),        # L leg
    (0, 7), (7, 8), (8, 9), (9, 10),  # spine → head
    (8, 11), (11, 12), (12, 13),   # L arm
    (8, 14), (14, 15), (15, 16),   # R arm
]

EVAL_JOINTS_14 = [1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16]

def norm_by_bone_length(pred, gt, bones=H36M_BONES):
    """Per-bone length normalization (GnTCN protocol).

    For each bone, rescale the predicted vector to match GT bone length
    while preserving the predicted direction.

    Args:
        pred: (..., J, 3) predicted 3D pose
        gt:   (..., J, 3) ground truth 3D pose (root-aligned)
    Returns:
        mapped: (..., J, 3) prediction with bone lengths matched to GT
    """
    mapped = pred.copy()
    for parent, child in bones:
        gt_vec = gt[..., child, :] - gt[..., parent, :]
        gt_len = np.linalg.norm(gt_vec, axis=-1, keepdims=True)

        pred_vec = pred[..., child, :] - pred[..., parent, :]
        pred_len = np.linalg.norm(pred_vec, axis=-1, keepdims=True)

        # Avoid division by zero; if pred_len ≈ 0, keep original
        scale = np.where(pred_len > 1e-6, gt_len / pred_len, 1.0)
        mapped[..., child, :] = mapped[..., parent, :] + pred_vec * scale
    return mapped
