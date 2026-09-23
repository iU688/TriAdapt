"""Paper-aligned root-relative pose metrics for source-only validation."""

from collections import defaultdict

import numpy as np

from source_pose_geometry import EVAL_JOINTS_14, norm_by_bone_length


def bone_normalized_errors(prediction, target):
    """Return MuPoTS-style 14-joint errors in mm for matched poses."""
    prediction = np.asarray(prediction, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if prediction.shape != target.shape or prediction.shape[-2:] != (17, 3):
        raise ValueError(f'expected matching (...,17,3) arrays, got {prediction.shape} and {target.shape}')
    prediction = prediction - prediction[..., 0:1, :]
    target = target - target[..., 0:1, :]
    normalized = norm_by_bone_length(prediction, target)
    return np.linalg.norm(
        normalized[..., EVAL_JOINTS_14, :] - target[..., EVAL_JOINTS_14, :],
        axis=-1,
    )


def summarize_sequence_errors(errors_by_sequence):
    """Pool joints/people per sequence, then average sequence metrics."""
    rows = []
    thresholds = np.arange(0.0, 151.0, 5.0)
    for sequence in sorted(errors_by_sequence):
        parts = errors_by_sequence[sequence]
        if not parts:
            continue
        errors = np.concatenate([np.asarray(part).reshape(-1) for part in parts])
        rows.append({
            'sequence': sequence,
            'PCK_rel': float((errors < 150.0).mean() * 100.0),
            'AUC_rel': float(np.mean([(errors < value).mean() for value in thresholds]) * 100.0),
            'MPJPE_rel': float(errors.mean()),
            'joints': int(errors.size),
        })
    if not rows:
        return {'PCK_rel': float('nan'), 'AUC_rel': float('nan'),
                'AUC_0_195': float('nan'),
                'MPJPE_rel': float('nan'), 'sequences': 0}
    result = {
        'PCK_rel': float(np.mean([row['PCK_rel'] for row in rows])),
        'AUC_rel': float(np.mean([row['AUC_rel'] for row in rows])),
        'MPJPE_rel': float(np.mean([row['MPJPE_rel'] for row in rows])),
        'sequences': len(rows),
    }
    # Compatibility alias for historical logs. The value now uses 0:5:150.
    result['AUC_0_195'] = result['AUC_rel']
    return result


class SourcePaperMetricAccumulator:
    def __init__(self):
        self.errors = defaultdict(list)

    def update(self, sequences, prediction, target):
        errors = bone_normalized_errors(prediction, target)
        if len(sequences) != errors.shape[0]:
            raise ValueError('sequence labels do not match batch size')
        for index, sequence in enumerate(sequences):
            self.errors[str(sequence)].append(errors[index])

    def summarize(self):
        return summarize_sequence_errors(self.errors)
