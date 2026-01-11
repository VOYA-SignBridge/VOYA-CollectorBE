"""oversample_balance.py
Execute augmentation plan produced by balancer.
For each target class, randomly sample existing sequences and generate extra augmentations until deficit closed.
"""
from __future__ import annotations

import random
import numpy as np
from pathlib import Path
from typing import List
from app.balancer import build_balance_plan
from app.dataset_manager import ClassMetadata, load_labels
from app.dataset_samples import save_sequence_npz
from app.processing.augmenter import generate_augmented_sequences

def get_class_dirs() -> dict[str, Path]:
    rows = load_labels()
    out = {}
    from app.dataset_manager import ClassMetadata
    for r in rows:
        meta = ClassMetadata(
            class_uid=r['class_uid'], slug=r['slug'], label_original=r['label_original'],
            language=r['language'], dialect=r['dialect'],
            is_common_global=bool(int(r['is_common_global'])),
            is_common_language=bool(int(r['is_common_language']))
        )
        out[meta.class_uid] = meta.hierarchy_path()
    return out

def load_sequences_for_class(class_uid: str, limit: int | None = None) -> List[np.ndarray]:
    dirs = get_class_dirs()
    path = dirs.get(class_uid)
    if not path or not path.exists():
        return []
    seqs = []
    for f in path.glob('*.npz'):
        try:
            with np.load(f, allow_pickle=True) as data:
                seq = data.get('sequence')
                if seq is not None and seq.ndim == 2:
                    seqs.append(seq.astype(np.float32))
        except Exception:
            continue
        if limit and len(seqs) >= limit:
            break
    return seqs

def execute(target: int | None = None, per_sequence_aug: int = 2):
    plan = build_balance_plan(target=target)
    dirs = get_class_dirs()
    rows = load_labels()
    uid_to_meta = {r['class_uid']: ClassMetadata(
        class_uid=r['class_uid'], slug=r['slug'], label_original=r['label_original'],
        language=r['language'], dialect=r['dialect'],
        is_common_global=bool(int(r['is_common_global'])),
        is_common_language=bool(int(r['is_common_language']))
    ) for r in rows}
    performed = []
    for item in plan.get('plan', []):
        cid = item['class_uid']
        deficit = item['deficit']
        meta = uid_to_meta.get(cid)
        if not meta:
            continue
        existing = load_sequences_for_class(cid)
        if not existing:
            continue
        # cycle through existing sequences
        idx = 0
        created = 0
        while created < deficit:
            base_seq = existing[idx % len(existing)]
            aug_list = generate_augmented_sequences(base_seq, config={'n': per_sequence_aug})
            for i, aseq in enumerate(aug_list):
                if created >= deficit:
                    break
                save_sequence_npz(meta, aseq, meta={'oversampled': True}, augment_id=i, source_type='oversample')
                created += 1
            idx += 1
        performed.append({'class_uid': cid, 'added': created})
    return {'performed': performed}

if __name__ == '__main__':
    print(execute())
