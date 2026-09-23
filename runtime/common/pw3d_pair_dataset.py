"""Pair-only 3DPW train/validation data for the isolated R2 pilot."""

from itertools import combinations
import json

import numpy as np
import torch
from torch.utils.data import Dataset


LEFT = [4, 5, 6, 11, 12, 13]
RIGHT = [1, 2, 3, 14, 15, 16]
BONES = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
         (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
         (12, 13), (8, 14), (14, 15), (15, 16)]


def normalize_screen(points, width, height):
    result = points.astype(np.float32, copy=True)
    visible = np.any(result != 0.0, axis=-1)
    result = result / float(width) * 2.0 - np.asarray([1.0, height / width], dtype=np.float32)
    result[~visible] = 0.0
    return result


def pose_anchor(points, confidence, width, height):
    visible = confidence > 0.0
    lengths = []
    for parent, child in BONES:
        if visible[parent] and visible[child]:
            lengths.append(np.linalg.norm(points[child] - points[parent]))
    scale = (float(np.mean(lengths)) if lengths else 1.0) / float(width) * 2.0
    pelvis = normalize_screen(points[None], width, height)[0, 0]
    return np.asarray([max(scale, 1e-6), pelvis[0], pelvis[1]], dtype=np.float32)


class PW3DPairDataset(Dataset):
    def __init__(self, path, frames=27, center_stride=1, augment_flip=False, seed=42,
                 include_single=False):
        super().__init__()
        source = np.load(path, allow_pickle=True)
        self.metadata = json.loads(str(source["metadata"].item()))
        if self.metadata.get("dataset") != "3DPW" or self.metadata.get("target_test_used") is not False:
            raise ValueError(f"{path}: R2 accepts only audited non-test 3DPW data")
        self.detector = source["positions_2d"].item()
        self.gt2d = source["positions_2d_gt"].item()
        self.scores = source["scores_2d"].item()
        self.targets = source["positions_3d"].item()
        self.valid = source["valid_person"].item()
        self.sizes = source["image_size"].item()
        self.frames = frames
        self.pad = frames // 2
        self.augment_flip = augment_flip
        self.seed = seed
        self.include_single = include_single
        self.epoch = 0
        self.samples = []
        for sequence in sorted(self.valid):
            valid = np.asarray(self.valid[sequence], dtype=bool)
            for center in range(0, len(valid), center_stride):
                people = np.flatnonzero(valid[center])
                pairs = list(combinations(people, 2))
                if include_single and len(people) == 1:
                    pairs = [(int(people[0]), int(people[0]))]
                self.samples.extend((sequence, center, pair) for pair in pairs)
        if not self.samples:
            raise ValueError(f"{path}: no valid two-person center frames")

    def __len__(self):
        return len(self.samples)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    @staticmethod
    def _flip_2d(value):
        result = value.copy()
        result[..., 0] *= -1.0
        result[..., LEFT + RIGHT, :] = result[..., RIGHT + LEFT, :]
        return result

    @staticmethod
    def _flip_confidence(value):
        result = value.copy()
        result[..., LEFT + RIGHT] = result[..., RIGHT + LEFT]
        return result

    @staticmethod
    def _flip_3d(value):
        result = value.copy()
        result[..., 0] *= -1.0
        result[..., LEFT + RIGHT, :] = result[..., RIGHT + LEFT, :]
        return result

    def __getitem__(self, index):
        sequence, center, pair = self.samples[index]
        indices = np.clip(np.arange(center - self.pad, center + self.pad + 1),
                          0, len(self.valid[sequence]) - 1)
        width, height = map(float, self.sizes[sequence])
        detector, gt2d, confidence, anchors = [], [], [], []
        for person in pair:
            temporal_valid = np.asarray(self.valid[sequence][indices, person], dtype=bool)
            det_pixels = np.asarray(self.detector[sequence][indices, person], dtype=np.float32).copy()
            gt_pixels = np.asarray(self.gt2d[sequence][indices, person], dtype=np.float32).copy()
            scores = np.asarray(self.scores[sequence][indices, person], dtype=np.float32).copy()
            det_pixels[~temporal_valid] = 0.0
            gt_pixels[~temporal_valid] = 0.0
            scores[~temporal_valid] = 0.0
            detector.append(normalize_screen(det_pixels, width, height))
            gt2d.append(normalize_screen(gt_pixels, width, height))
            confidence.append(np.clip(scores, 0.0, 1.0))
            anchors.append(pose_anchor(det_pixels[self.pad], scores[self.pad], width, height))
        detector = np.stack(detector)
        gt2d = np.stack(gt2d)
        confidence = np.stack(confidence)
        gt_confidence = np.stack([
            np.repeat(np.asarray(self.valid[sequence][indices, person], dtype=np.float32)[:, None], 17, axis=1)
            for person in pair
        ])
        target = np.asarray(self.targets[sequence][center, list(pair)], dtype=np.float32)
        anchors = np.stack(anchors)

        flip_seed = self.seed + self.epoch * max(len(self.samples), 1) + index
        if self.augment_flip and np.random.RandomState(flip_seed).rand() < 0.5:
            detector = self._flip_2d(detector)
            gt2d = self._flip_2d(gt2d)
            confidence = self._flip_confidence(confidence)
            gt_confidence = self._flip_confidence(gt_confidence)
            target = self._flip_3d(target)
            anchors[:, 1] *= -1.0

        return {
            "detector": torch.from_numpy(detector),
            "gt2d": torch.from_numpy(gt2d),
            "confidence": torch.from_numpy(confidence),
            "gt_confidence": torch.from_numpy(gt_confidence),
            "target": torch.from_numpy(target),
            "anchors": torch.from_numpy(anchors),
            "sequence": sequence,
            "center": torch.tensor(center, dtype=torch.long),
            "pair": torch.tensor(pair, dtype=torch.long),
            "source": "pw3d",
            "pair_is_distinct": torch.tensor(pair[0] != pair[1], dtype=torch.bool),
        }
