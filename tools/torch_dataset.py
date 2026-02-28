import csv
import os
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_sequence_from_npz(path: str) -> Tuple[np.ndarray, Dict]:
    data = np.load(path, allow_pickle=True)
    # try to get array named 'sequence' else first array
    if 'sequence' in data:
        seq = data['sequence']
    else:
        keys = list(data.keys())
        seq = data[keys[0]]

    meta: Dict = {}
    if 'meta' in data:
        try:
            m = data['meta']
            # meta may be a numpy object array
            if isinstance(m, np.ndarray) and m.shape == ():
                meta = dict(m.item())
            elif isinstance(m, dict):
                meta = dict(m)
        except Exception:
            meta = {}

    return seq.astype(np.float32), meta


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
    """
    Mirror sequence for data augmentation (hands-only format)
    Expected format: 126 dimensions = left_hand (63) + right_hand (63)
    Each hand: 21 landmarks × 3 coordinates (x,y,z)
    """
    if seq.ndim != 2 or seq.shape[1] < 126:
        # Not a hands-only representation; skip mirroring.
        return seq
    m = seq.copy()
    
    # Mirror X coordinates for hands (every 3rd element starting at 0)
    # Left hand: indices 0 to 62 (step 3 for x coords)
    for x_idx in range(0, 63, 3):
        m[:, x_idx] = -m[:, x_idx]
    
    # Right hand: indices 63 to 125 (step 3 for x coords)  
    for x_idx in range(63, 126, 3):
        m[:, x_idx] = -m[:, x_idx]
    
    # Swap left and right hand blocks
    left_hand = m[:, 0:63].copy()    # First 63 dims
    right_hand = m[:, 63:126].copy() # Next 63 dims
    
    m[:, 0:63] = right_hand    # Put right hand in left position
    m[:, 63:126] = left_hand   # Put left hand in right position
    
    return m


@dataclass
class SampleInfo:
    npz_path: str
    label: int
    class_uid: str
    slug: str
    label_original: str
    user: str
    session_id: str


def _parse_class_dir_name(name: str) -> Tuple[str, str]:
    """Parse '<class_uid>_<slug>' -> (class_uid, slug).

    The current dataset uses folder names like: 'class_0003_xin-chao'.
    In that case, class_uid should be 'class_0003' and slug 'xin-chao'.
    """
    if '_' not in name:
        return name, ""

    parts = name.split('_')
    if len(parts) >= 3 and parts[0] == 'class' and parts[1].isdigit():
        class_uid = f"class_{parts[1]}"
        slug = '_'.join(parts[2:])
        return class_uid, slug

    class_uid, slug = name.split('_', 1)
    return class_uid, slug


def _read_labels_csv(path: Path) -> Dict[str, Dict[str, str]]:
    """Index labels.csv rows by multiple keys.

    Historically we used class_uid, but in the current dataset `class_uid` is a UUID,
    while feature folders are named like 'class_0003_xin-chao' (stored in `folder_name`).
    We therefore index each row by both `class_uid` and `folder_name` when present.
    """
    if not path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cu = (row.get('class_uid') or '').strip()
            fn = (row.get('folder_name') or '').strip()
            lang = (row.get('language') or '').strip()
            dia = (row.get('dialect') or '').strip()
            if not cu and not fn:
                continue
            normalized = {k: (v if v is not None else '') for k, v in row.items()}
            # Prefer a fully-qualified key when folder names are reused across dialects.
            if fn and lang and dia:
                out[f"{fn}|{lang}|{dia}"] = normalized
            if cu:
                out[cu] = normalized
            if fn:
                out[fn] = normalized
    return out


class SignDataset(Dataset):
    """Dataset loader for the current dataset hierarchy.

    Supports:
    - New hierarchy: dataset/features/<language>/<dialect>/<class_uid>_<slug>/sample_*.npz
    - Optional fallback to a flat directory of class folders.

    Also supports:
    - On-the-fly augmentation
    - Optional per-dim normalization (mean/std)
    - Group metadata (user/session) for leakage-safe splits
    """

    def __init__(
        self,
        features_root: str = 'dataset/features',
        labels_csv: str = 'dataset/labels.csv',
        language: str = 'vn',
        dialect: str = 'common',
        augment: bool = True,
        max_samples: Optional[int] = None,
        expected_dim: Optional[int] = 126,
    ):
        self.features_root = Path(features_root)
        self.labels_csv = Path(labels_csv)
        self.language = language
        self.dialect = dialect
        self.augment = augment
        self.expected_dim = expected_dim

        self.samples: List[SampleInfo] = []
        self.norm_mean: Optional[np.ndarray] = None
        self.norm_std: Optional[np.ndarray] = None
        self.input_dim: Optional[int] = None

        self._labels_by_uid = _read_labels_csv(self.labels_csv)
        self._load_samples(max_samples)

    def set_normalization(self, mean: Optional[np.ndarray], std: Optional[np.ndarray]) -> None:
        self.norm_mean = mean
        self.norm_std = std

    def _iter_class_dirs(self) -> List[Path]:
        # New hierarchy
        lang_dir = self.features_root / self.language
        if lang_dir.exists() and lang_dir.is_dir():
            d = (self.dialect or '').strip().lower()
            if d in {'*', 'all', 'any'}:
                # Merge all dialect folders under the language.
                # Order is stable: dialect folder name, then class folder name.
                out: List[Path] = []
                for dialect_dir in sorted([p for p in lang_dir.iterdir() if p.is_dir()]):
                    out.extend(sorted([p for p in dialect_dir.iterdir() if p.is_dir()]))
                return out

            dialect_dir = lang_dir / self.dialect
            if dialect_dir.exists() and dialect_dir.is_dir():
                return sorted([p for p in dialect_dir.iterdir() if p.is_dir()])
            return []

        # Fallback: flat
        if self.features_root.exists() and self.features_root.is_dir():
            return sorted([p for p in self.features_root.iterdir() if p.is_dir()])
        return []

    def _load_samples(self, max_samples: Optional[int]) -> None:
        class_dirs = self._iter_class_dirs()
        for class_dir in class_dirs:
            # When iterating across dialects/languages, infer them from the hierarchy.
            inferred_dialect = class_dir.parent.name
            inferred_language = class_dir.parent.parent.name if class_dir.parent.parent is not None else ''

            class_uid, slug_from_dir = _parse_class_dir_name(class_dir.name)
            # labels.csv may use UUID `class_uid`, but stores the folder name in `folder_name`.
            # Prefer matching by folder_name (exact dir name), then by parsed class_uid.
            fq = f"{class_dir.name}|{inferred_language}|{inferred_dialect}"
            row = (
                self._labels_by_uid.get(fq, {})
                or self._labels_by_uid.get(class_dir.name, {})
                or self._labels_by_uid.get(class_uid, {})
            )

            # class_idx is our canonical class id (not necessarily contiguous)
            label_str = (row.get('class_idx') or '').strip()
            if label_str:
                try:
                    label = int(label_str)
                except Exception:
                    label = -1
            else:
                label = -1

            slug = (row.get('slug') or slug_from_dir or '').strip()
            label_original = (row.get('label_original') or '').strip()
            true_class_uid = (row.get('class_uid') or '').strip() or str(class_uid)

            for npz_path in sorted(class_dir.glob('*.npz')):
                # sidecar JSON
                sidecar = npz_path.with_suffix('.json')
                user = ''
                session_id = ''
                if sidecar.exists():
                    try:
                        md = json.loads(sidecar.read_text(encoding='utf-8'))
                        user = str(md.get('user') or '')
                        session_id = str(md.get('session_id') or '')
                    except Exception:
                        user = ''
                        session_id = ''
                else:
                    # fallback: try meta inside npz
                    try:
                        _, meta = _load_sequence_from_npz(str(npz_path))
                        user = str(meta.get('user') or '')
                        session_id = str(meta.get('session_id') or '')
                    except Exception:
                        pass

                self.samples.append(
                    SampleInfo(
                        npz_path=str(npz_path),
                        label=int(label),
                        class_uid=str(true_class_uid),
                        slug=str(slug),
                        label_original=str(label_original),
                        user=user,
                        session_id=session_id,
                    )
                )
                if max_samples and len(self.samples) >= int(max_samples):
                    break
            if max_samples and len(self.samples) >= int(max_samples):
                break

        # Infer input dim from first valid sample
        for s in self.samples:
            try:
                seq, _ = _load_sequence_from_npz(s.npz_path)
                if isinstance(seq, np.ndarray) and seq.ndim == 2:
                    self.input_dim = int(seq.shape[1])
                    break
            except Exception:
                continue

        # If we coerce to expected_dim, treat that as the effective input_dim
        if self.expected_dim is not None:
            self.input_dim = int(self.expected_dim)

    def _coerce_dim(self, seq: np.ndarray) -> np.ndarray:
        if seq.ndim != 2:
            return seq
        if self.expected_dim is None:
            return seq

        D = int(seq.shape[1])
        target = int(self.expected_dim)
        if D == target:
            return seq
        if D > target:
            return seq[:, :target].astype(np.float32)
        # pad
        pad = np.zeros((seq.shape[0], target - D), dtype=np.float32)
        return np.concatenate([seq.astype(np.float32), pad], axis=1)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        seq, _meta = _load_sequence_from_npz(s.npz_path)
        seq = self._coerce_dim(seq)

        # track inferred dim
        if self.input_dim is None and isinstance(seq, np.ndarray) and seq.ndim == 2:
            self.input_dim = int(seq.shape[1])

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

        # optional normalization
        if self.norm_mean is not None and self.norm_std is not None:
            try:
                m = self.norm_mean
                st = self.norm_std
                # broadcast over time
                seq = (seq - m) / st
            except Exception:
                pass

        # ensure float32 and shape
        seq = seq.astype(np.float32)
        # return tensor: (T, D)
        group = s.session_id or s.user or ""
        return torch.from_numpy(seq), int(s.label), group
