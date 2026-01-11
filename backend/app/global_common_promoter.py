"""global_common_promoter.py
Identify gestures appearing across multiple languages and optionally promote them
into a single global_common class. Copies sequences into global_common directory.
Existing language-specific classes remain (backward compatibility) unless --prune specified.
"""
from __future__ import annotations

import argparse, shutil
from pathlib import Path
import numpy as np
from collections import defaultdict

from app.dataset_manager import load_labels, register_class, ClassMetadata

def find_multi_language_slugs(min_languages: int = 2):
    rows = load_labels()
    slug_map = defaultdict(set)
    meta_map = defaultdict(list)
    for r in rows:
        if int(r['is_common_global']) == 1:
            continue
        slug_map[r['slug']].add(r['language'])
        meta_map[r['slug']].append(r)
    candidates = []
    for slug, langs in slug_map.items():
        if len(langs) >= min_languages:
            candidates.append((slug, sorted(list(langs)), meta_map[slug]))
    return candidates

def promote(slug: str, metas: list[dict], dry_run: bool = False):
    # choose representative label_original (first)
    label_original = metas[0]['label_original']
    global_meta = register_class(label_original=label_original, language='global', dialect='global', is_common_global=True)
    target_dir = global_meta.hierarchy_path()
    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for m in metas:
        src_meta = ClassMetadata(
            class_uid=m['class_uid'], slug=m['slug'], label_original=m['label_original'],
            language=m['language'], dialect=m['dialect'],
            is_common_global=False, is_common_language=bool(int(m['is_common_language']))
        )
        src_dir = src_meta.hierarchy_path()
        if not src_dir.exists():
            continue
        for f in src_dir.glob('*.npz'):
            if dry_run:
                copied += 1
                continue
            # Copy keeping original filename but prefix with source language
            new_name = f"{m['language']}_{f.name}"
            shutil.copy2(f, target_dir / new_name)
            copied += 1
    return {'global_class_uid': global_meta.class_uid, 'copied': copied, 'slug': slug}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-languages', type=int, default=2)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--promote', action='store_true', help='Execute promotion copy')
    args = ap.parse_args()
    candidates = find_multi_language_slugs(min_languages=args.min_languages)
    out = []
    for slug, langs, metas in candidates:
        if args.promote:
            res = promote(slug, metas, dry_run=args.dry_run)
            out.append(res)
        else:
            out.append({'slug': slug, 'languages': langs, 'count': sum(1 for _ in metas)})
    print({'candidates': out, 'dry_run': args.dry_run, 'promoted': args.promote})

if __name__ == '__main__':
    main()
