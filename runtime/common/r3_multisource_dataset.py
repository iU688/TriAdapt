"""Screen-normalized MPI pair and Human3.6M replay datasets for R3."""

import bisect

import numpy as np
import torch
from torch.utils.data import Dataset

from common.pw3d_pair_dataset import LEFT, RIGHT, normalize_screen, pose_anchor


def contiguous_runs(frame_ids):
    frame_ids = sorted(map(int, frame_ids))
    if not frame_ids:
        return []
    runs, run = [], [frame_ids[0]]
    for frame in frame_ids[1:]:
        if frame == run[-1] + 1:
            run.append(frame)
        else:
            runs.append(run)
            run = [frame]
    runs.append(run)
    return runs


def flip_2d(value):
    result = value.copy()
    result[..., 0] *= -1.0
    result[..., LEFT + RIGHT, :] = result[..., RIGHT + LEFT, :]
    return result


def flip_confidence(value):
    result = value.copy()
    result[..., LEFT + RIGHT] = result[..., RIGHT + LEFT]
    return result


def flip_3d(value):
    result = value.copy()
    result[..., 0] *= -1.0
    result[..., LEFT + RIGHT, :] = result[..., RIGHT + LEFT, :]
    return result


class MPIWeakPairDataset(Dataset):
    def __init__(self, path, frames=27, augment_flip=True, seed=42):
        source = np.load(path, allow_pickle=True)
        self.p2d = source["positions_2d"].item()
        self.p3d = source["positions_3d"].item()
        self.valids = source["valids"].item()
        self.config = source["generation_config"].item()
        self.width = float(self.config["img_w"])
        self.height = float(self.config["img_h"])
        self.frames = frames
        self.pad = frames // 2
        self.augment_flip = augment_flip
        self.seed = seed
        self.epoch = 0
        self.runs = []
        lengths = []
        for sequence in sorted(self.p2d):
            pair_frames = [frame for frame, people in self.p2d[sequence].items()
                           if len(people) >= 2]
            for run in contiguous_runs(pair_frames):
                self.runs.append((sequence, np.asarray(run, dtype=np.int64)))
                lengths.append(len(run))
        self.cumulative = np.cumsum(lengths).tolist()
        if not self.cumulative:
            raise ValueError(f"{path}: no two-person temporal runs")

    def __len__(self):
        return self.cumulative[-1]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        run_index = bisect.bisect_right(self.cumulative, index)
        previous = 0 if run_index == 0 else self.cumulative[run_index - 1]
        position = index - previous
        sequence, run = self.runs[run_index]
        center = int(run[position])
        positions = np.clip(np.arange(position - self.pad, position + self.pad + 1),
                            0, len(run) - 1)
        frame_ids = run[positions]
        people = sorted(self.p2d[sequence][center])[:2]
        detector, gt2d, confidence, anchors = [], [], [], []
        for person in people:
            pixels = np.stack([self.p2d[sequence][int(frame)][person] for frame in frame_ids]).astype(np.float32)
            masks = np.stack([self.valids[sequence][int(frame)][person] for frame in frame_ids]).astype(bool)
            observed = pixels.copy()
            observed[~masks] = 0.0
            detector.append(normalize_screen(observed, self.width, self.height))
            gt2d.append(normalize_screen(pixels, self.width, self.height))
            confidence.append(masks.astype(np.float32))
            anchors.append(pose_anchor(pixels[self.pad], masks[self.pad], self.width, self.height))
        detector, gt2d = np.stack(detector), np.stack(gt2d)
        confidence = np.stack(confidence)
        gt_confidence = np.ones_like(confidence, dtype=np.float32)
        target = np.stack([self.p3d[sequence][center][person] for person in people]).astype(np.float32)
        anchors = np.stack(anchors)
        flip_seed = self.seed + self.epoch * len(self) + index
        if self.augment_flip and np.random.RandomState(flip_seed).rand() < 0.5:
            detector, gt2d = flip_2d(detector), flip_2d(gt2d)
            confidence, gt_confidence = flip_confidence(confidence), flip_confidence(gt_confidence)
            target = flip_3d(target)
            anchors[:, 1] *= -1.0
        return {
            "detector": torch.from_numpy(detector), "gt2d": torch.from_numpy(gt2d),
            "confidence": torch.from_numpy(confidence),
            "gt_confidence": torch.from_numpy(gt_confidence),
            "target": torch.from_numpy(target), "anchors": torch.from_numpy(anchors),
            "sequence": sequence,
            "center": torch.tensor(center, dtype=torch.long),
            "pair": torch.tensor(people, dtype=torch.long),
            "source": "mpi",
            "pair_is_distinct": torch.tensor(True, dtype=torch.bool),
        }


class H36MReplayDataset(Dataset):
    RESOLUTIONS = ((1000.0, 1002.0), (1000.0, 1000.0),
                   (1000.0, 1000.0), (1000.0, 1002.0))

    def __init__(self, path, frames=27, subjects=("S1", "S5", "S6", "S7", "S8"),
                 augment_flip=True, seed=42):
        positions = np.load(path, allow_pickle=True)["positions_2d"].item()
        self.sequences, lengths = [], []
        for subject in subjects:
            for action in sorted(positions[subject]):
                for camera_index, array in enumerate(positions[subject][action]):
                    self.sequences.append((f"{subject}/{action}/cam{camera_index}",
                                           np.asarray(array[..., :2], dtype=np.float32), camera_index))
                    lengths.append(len(array))
        self.cumulative = np.cumsum(lengths).tolist()
        self.frames = frames
        self.pad = frames // 2
        self.augment_flip = augment_flip
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return self.cumulative[-1]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __getitem__(self, index):
        sequence_index = bisect.bisect_right(self.cumulative, index)
        previous = 0 if sequence_index == 0 else self.cumulative[sequence_index - 1]
        center = index - previous
        name, array, camera = self.sequences[sequence_index]
        indices = np.clip(np.arange(center - self.pad, center + self.pad + 1), 0, len(array) - 1)
        pixels = array[indices].copy()
        width, height = self.RESOLUTIONS[camera]
        normalized = normalize_screen(pixels, width, height)
        confidence = np.ones((self.frames, 17), dtype=np.float32)
        anchor = pose_anchor(pixels[self.pad], confidence[self.pad], width, height)
        dual = np.stack([normalized, normalized])
        dual_confidence = np.stack([confidence, confidence])
        anchors = np.stack([anchor, anchor])
        flip_seed = self.seed + self.epoch * len(self) + index
        if self.augment_flip and np.random.RandomState(flip_seed).rand() < 0.5:
            dual = flip_2d(dual)
            dual_confidence = flip_confidence(dual_confidence)
            anchors[:, 1] *= -1.0
        return {
            "detector": torch.from_numpy(dual), "gt2d": torch.from_numpy(dual.copy()),
            "confidence": torch.from_numpy(dual_confidence),
            "gt_confidence": torch.from_numpy(dual_confidence.copy()),
            "target": torch.zeros((2, 17, 3), dtype=torch.float32),
            "anchors": torch.from_numpy(anchors), "sequence": name, "source": "h36m",
            "pair_is_distinct": torch.tensor(False, dtype=torch.bool),
        }
