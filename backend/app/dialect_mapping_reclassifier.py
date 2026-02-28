"""dialect_mapping_reclassifier.py
CSV-driven dialect reassignment for existing classes.

Mapping CSV columns (required header):
  label_original,dialect
Optional columns: language (defaults 'vn'), slug

Dry run example:
  python -m app.dialect_mapping_reclassifier --mapping dialect_mapping.csv --dry-run

Apply example (move files & update CSVs):
  python -m app.dialect_mapping_reclassifier --mapping dialect_mapping.csv --apply --remove-old

Notes:
 - Skips rows without label_original or dialect.
 - Ignores rows where target dialect equals current dialect.
 - Does not assign 'common' targets via this script (skip).
 - Moves all samples of a class; does not split.
 - Updates samples.csv (class_uid, dialect, file_path) atomically.
 - Regenerates label indexes after modification.
"""
from __future__ import annotations

import csv
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from app.dataset_manager import load_labels, register_class, ClassMetadata, MASTER_LABELS, LABEL_FIELDS, regenerate_label_indexes
from app.dataset_samples import SAMPLES_CSV, SAMPLE_FIELDS
from filelock import FileLock

def _resolve_mapping_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    # Try relative to current working directory
    p2 = Path.cwd() / path
    if p2.exists():
        return p2
    # Try project root one level up (backend -> project)
    p3 = Path.cwd().parent / path
    if p3.exists():
        return p3
    return p

def read_mapping(path: str) -> List[Dict[str,str]]:
    rows: List[Dict[str,str]] = []
    resolved = _resolve_mapping_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Mapping CSV not found: {path} (resolved: {resolved})")
    with open(resolved, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

def load_samples() -> List[Dict[str,str]]:
    if not Path(SAMPLES_CSV).exists():
        return []
    with open(SAMPLES_CSV, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def write_samples(rows: List[Dict[str,str]]):
    lock = FileLock(str(SAMPLES_CSV) + '.lock')
    with lock:
        with open(SAMPLES_CSV, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
            w.writeheader(); w.writerows(rows)

def write_labels(rows: List[Dict[str,str]]):
    lock = FileLock(str(MASTER_LABELS) + '.lock')
    with lock:
        with open(MASTER_LABELS, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
            w.writeheader(); w.writerows(rows)
    regenerate_label_indexes()

def find_label(rows: List[Dict[str,str]], language: str, label_original: str, slug: Optional[str]) -> Optional[Dict[str,str]]:
    for r in rows:
        if r['language'] == language and r['label_original'] == label_original:
            return r
    if slug:
        for r in rows:
            if r['language'] == language and r['slug'] == slug:
                return r
    return None

def reclassify(mapping_csv: str, apply: bool = False, remove_old: bool = False, default_language: str = 'vn') -> Dict:
    label_rows = load_labels()
    samples = load_samples()
    mapping = read_mapping(mapping_csv)
    changes = []
    updated_samples = samples[:]
    new_label_rows = label_rows[:]

    for m in mapping:
        label_original = (m.get('label_original') or '').strip()
        if not label_original:
            continue
        target_dialect = (m.get('dialect') or '').strip().lower()
        if not target_dialect:
            continue
        if target_dialect == 'common':  # skip: handled separately by register as language common
            changes.append({'label_original': label_original, 'status': 'skip_common_target'})
            continue
        language = (m.get('language') or default_language).strip().lower()
        slug = (m.get('slug') or '').strip() or None
        existing = find_label(label_rows, language, label_original, slug)
        if not existing:
            changes.append({'label_original': label_original, 'status': 'not_found'})
            continue
        current_dialect = existing['dialect']
        if current_dialect == target_dialect:
            changes.append({'label_original': label_original, 'status': 'already_correct', 'dialect': current_dialect})
            continue

        # register new dialect class
        new_meta = register_class(label_original=label_original, language=language, dialect=target_dialect)
        source_meta = ClassMetadata(
            class_uid=existing['class_uid'], slug=existing['slug'], label_original=existing['label_original'],
            language=existing['language'], dialect=existing['dialect'],
            is_common_global=bool(int(existing['is_common_global'])),
            is_common_language=bool(int(existing['is_common_language']))
        )
        src_dir = source_meta.hierarchy_path()
        dst_dir = new_meta.hierarchy_path()
        moved = 0
        if apply and src_dir.exists():
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.glob('*.npz'):
                shutil.move(str(f), dst_dir / f.name)
                moved += 1
            for j in src_dir.glob('*.json'):
                shutil.move(str(j), dst_dir / j.name)
        # update samples entries
        if apply:
            for s in updated_samples:
                if s.get('class_uid') == source_meta.class_uid:
                    s['class_uid'] = new_meta.class_uid
                    s['dialect'] = target_dialect
                    # update file_path if points to old directory
                    fp = s.get('file_path','')
                    if fp and source_meta.class_uid in fp:
                        old_path = Path(fp)
                        new_path = dst_dir / old_path.name
                        s['file_path'] = str(new_path)
        # update labels_master
        if apply and remove_old:
            new_label_rows = [r for r in new_label_rows if r['class_uid'] != source_meta.class_uid]
        changes.append({'label_original': label_original, 'from': current_dialect, 'to': target_dialect, 'moved': moved, 'new_class_uid': new_meta.class_uid})

    if apply:
        write_samples(updated_samples)
        write_labels(new_label_rows)
    return {'changes': changes, 'applied': apply, 'remove_old': remove_old}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapping', required=True, help='Path to dialect mapping CSV')
    ap.add_argument('--apply', action='store_true', help='Apply changes (move files & update CSVs)')
    ap.add_argument('--remove-old', action='store_true', help='Remove old label rows after move')
    ap.add_argument('--default-language', default='vn')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    res = reclassify(mapping_csv=args.mapping, apply=args.apply and not args.dry_run, remove_old=args.remove_old, default_language=args.default_language)
    print(res)

if __name__ == '__main__':
    main()