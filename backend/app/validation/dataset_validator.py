"""dataset_validator.py
Validate multilingual hierarchical dataset integrity.

Checks:
 - Counts per class vs files present
 - Sequence shape consistency (expected seq_len x feature_dim)
 - Completeness metadata distribution
 - Missing or malformed metadata fields
 - Duplicate sample detection (same augment_id + start_frame + end_frame + class_uid)

Run examples:
  python -m app.validation.dataset_validator --language vn --dialect common
  python -m app.validation.dataset_validator --all
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

from app.dataset_manager import FEATURES_ROOT, load_labels, ClassMetadata
from app.config import settings

SEQ_LEN = int(getattr(settings, 'seq_len', 60))
FEATURE_DIM = int(getattr(settings, 'feature_dim', 126))


def load_class_meta(language: str | None, dialect: str | None):
    rows = load_labels()
    out = []
    for r in rows:
        if language and r['language'] != language:
            continue
        if dialect and r['dialect'] != dialect:
            continue
        out.append(ClassMetadata(
            class_uid=r['class_uid'],
            slug=r['slug'],
            label_original=r['label_original'],
            language=r['language'],
            dialect=r['dialect'],
            is_common_global=bool(int(r['is_common_global'])),
            is_common_language=bool(int(r['is_common_language'])),
        ))
    return out


def validate_class(meta: ClassMetadata):
    folder = meta.hierarchy_path()
    if not folder.exists():
        return {"class_uid": meta.class_uid, "slug": meta.slug, "count": 0, "missing_folder": True}

    npz_files = [p for p in folder.glob('*.npz')]
    stats = {
        "class_uid": meta.class_uid,
        "slug": meta.slug,
        "label": meta.label_original,
        "language": meta.language,
        "dialect": meta.dialect,
        "count": len(npz_files),
        "bad_shape": 0,
        "bad_meta": 0,
        "completeness": [],
        "duplicates": 0,
    }
    seen_keys = set()
    for f in npz_files:
        try:
            with np.load(f, allow_pickle=True) as data:
                seq = data.get('sequence')
                meta_dict = data.get('meta')
                if isinstance(meta_dict, np.ndarray) and meta_dict.dtype == object and meta_dict.shape == ():
                    meta_dict = dict(meta_dict.item())
                if seq is None or seq.ndim != 2:
                    stats['bad_shape'] += 1
                else:
                    # shape normalization check
                    t, d = seq.shape
                    if t != SEQ_LEN or d != FEATURE_DIM:
                        stats['bad_shape'] += 1
                if not isinstance(meta_dict, dict):
                    stats['bad_meta'] += 1
                    continue
                comp = meta_dict.get('completeness')
                if comp is not None:
                    try:
                        stats['completeness'].append(float(comp))
                    except Exception:
                        pass
                # Duplicate detection is meaningful mainly for video windows where
                # start/end frames are present; for live capture these are often missing.
                start_f = meta_dict.get('start_frame')
                end_f = meta_dict.get('end_frame')
                if start_f is not None and end_f is not None:
                    key = (
                        meta_dict.get('augment_id'),
                        meta_dict.get('speed_factor'),
                        start_f,
                        end_f,
                        meta.class_uid,
                    )
                    if key in seen_keys:
                        stats['duplicates'] += 1
                    else:
                        seen_keys.add(key)
        except Exception:
            stats['bad_meta'] += 1
    return stats


def summarize(all_stats):
    total = sum(s['count'] for s in all_stats)
    bad_shape = sum(s['bad_shape'] for s in all_stats)
    bad_meta = sum(s['bad_meta'] for s in all_stats)
    dups = sum(s['duplicates'] for s in all_stats)
    completeness_values = [c for s in all_stats for c in s['completeness']]
    comp_summary = {}
    if completeness_values:
        arr = np.array(completeness_values)
        comp_summary = {
            'mean': round(float(arr.mean()), 4),
            'min': round(float(arr.min()), 4),
            'p25': round(float(np.percentile(arr, 25)), 4),
            'median': round(float(np.percentile(arr, 50)), 4),
            'p75': round(float(np.percentile(arr, 75)), 4),
            'max': round(float(arr.max()), 4),
        }
    class_counts = {f"{s['language']}/{s['dialect']}:{s['slug']}": s['count'] for s in all_stats}
    # imbalance ratios
    max_count = max(class_counts.values()) if class_counts else 0
    imbalance = {k: round((v / max_count) if max_count else 0.0,4) for k,v in class_counts.items()}
    # dialect confusion: same slug across dialects of same language with very similar counts
    dialect_groups = defaultdict(list)
    for s in all_stats:
        dialect_groups[(s['language'], s['slug'])].append((s['dialect'], s['count']))
    confusion = []

    # cross-language collisions: identical slug across multiple languages
    cross_lang = defaultdict(set)
    for s in all_stats:
        cross_lang[s['slug']].add(s['language'])
    collisions = [
        {'slug': slug, 'languages': sorted(list(langs))}
        for slug, langs in cross_lang.items()
        if len(langs) > 1
    ]
    for (lang, slug), items in dialect_groups.items():
        if len(items) > 1:
            counts = [c for _, c in items]
            if max(counts) - min(counts) <= 2 and max(counts) > 0:  # heuristic
                confusion.append({'language': lang, 'slug': slug, 'dialect_counts': items})
    return {
        'total_samples': total,
        'classes': len(all_stats),
        'bad_shape_samples': bad_shape,
        'bad_meta_samples': bad_meta,
        'duplicate_sequences': dups,
        'completeness_summary': comp_summary,
        'per_class_counts': class_counts,
        'imbalance_ratios': imbalance,
        'dialect_confusion': confusion,
        'cross_language_collisions': collisions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', default=None)
    parser.add_argument('--dialect', default=None)
    parser.add_argument('--all', action='store_true', help='Ignore language/dialect filters')
    args = parser.parse_args()

    if args.all:
        metas = load_class_meta(None, None)
    else:
        metas = load_class_meta(args.language, args.dialect)
    if not metas:
        print({'error': 'No classes found for given filters', 'language': args.language, 'dialect': args.dialect})
        return
    stats = [validate_class(m) for m in metas]
    summary = summarize(stats)
    print(summary)

    # Optional verbose per-class anomalies
    anomalies = [s for s in stats if s['bad_shape'] or s['bad_meta'] or s['duplicates']]
    if anomalies:
        print({'anomalies': anomalies})


if __name__ == '__main__':
    main()
