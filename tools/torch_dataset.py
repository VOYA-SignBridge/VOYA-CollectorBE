import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset


def _load_sequence_from_npz(path):
    data = np.load(path)
    # try to get array named 'sequence' else first array
    if 'sequence' in data:
        seq = data['sequence']
    else:
        keys = list(data.keys())
        seq = data[keys[0]]
    return seq.astype(np.float32)


def jitter(seq, sigma=0.02):
    return seq + np.random.normal(0, sigma, seq.shape).astype(np.float32)


def scale(seq, factor=None, low=0.9, high=1.1):
    if factor is None:
        factor = np.random.uniform(low, high)
    return seq * float(factor)


def time_warp_resample(seq, factor=None, low=0.8, high=1.2):
    if factor is None:
        factor = float(np.random.uniform(low, high))
    T, D = seq.shape
    if factor == 1.0:
        return seq
    new_T = max(1, int(round(T * factor)))
    t_orig = np.arange(T)
    t_warp = np.linspace(0, T - 1, new_T)
    warped = np.zeros((new_T, D), dtype=np.float32)
    for d in range(D):
        warped[:, d] = np.interp(t_warp, t_orig, seq[:, d])
    t_resample = np.linspace(0, new_T - 1, T)
    resampled = np.zeros((T, D), dtype=np.float32)
    t_warp_indices = np.arange(new_T)
    for d in range(D):
        resampled[:, d] = np.interp(t_resample, t_warp_indices, warped[:, d])
    return resampled


def mirror_sequence(seq):
    m = seq.copy()
    # flip X for pose (first 100 entries, x every 4 starting at 0)
    for x_idx in range(0, 100, 4):
        m[:, x_idx] = -m[:, x_idx]
    # hands
    for x_idx in range(100, 100 + 63, 3):
        m[:, x_idx] = -m[:, x_idx]
    for x_idx in range(163, 163 + 63, 3):
        m[:, x_idx] = -m[:, x_idx]
    # swap hand blocks
    left = m[:, 100:163].copy()
    right = m[:, 163:226].copy()
    m[:, 100:163] = right
    m[:, 163:226] = left
    return m


class SignDataset(Dataset):
    """Dataset that loads .npz samples from dataset/features and applies on-the-fly augmentation."""

    def __init__(self, features_root='dataset/features', split_users=None, augment=True, max_samples=None):
        self.features_root = features_root
        self.samples = []  # list of tuples (npz_path, label_int, user)
        self.augment = augment
        self._load_samples(max_samples)

    def _load_samples(self, max_samples):
        classes = sorted(os.listdir(self.features_root))
        for cls_idx, cls in enumerate(classes, start=1):
            cls_path = os.path.join(self.features_root, cls)
            if not os.path.isdir(cls_path):
                continue
            for fname in os.listdir(cls_path):
                if not fname.endswith('.npz'):
                    continue
                fpath = os.path.join(cls_path, fname)
                # try to read metadata json alongside if exists
                meta_path = fpath[:-4] + '.json'
                user = None
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, 'r', encoding='utf-8') as fh:
                            md = json.load(fh)
                            user = md.get('user')
                    except Exception:
                        user = None
                self.samples.append((fpath, cls_idx - 1, user))
                if max_samples and len(self.samples) >= max_samples:
                    return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, user = self.samples[idx]
        seq = _load_sequence_from_npz(path)
        if self.augment:
            # apply random augmentations with probabilities
            if random.random() < 0.5:
                seq = jitter(seq, sigma=0.02)
            if random.random() < 0.3:
                seq = scale(seq)
            if random.random() < 0.3:
                seq = time_warp_resample(seq)
            if random.random() < 0.5:
                seq = mirror_sequence(seq)

        # ensure float32 and shape
        seq = seq.astype(np.float32)
        # return tensor: (T, D)
        return torch.from_numpy(seq), int(label), user
