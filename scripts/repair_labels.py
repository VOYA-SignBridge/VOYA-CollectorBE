"""Repair / inspect labels and feature folders.

Usage:
  # Report gaps and missing folders
  python scripts/repair_labels.py --report

  # Create missing feature folders (safe)
  python scripts/repair_labels.py --create-folders

Note: This script does NOT change class_idx numbers or rename folders.
If you want to compact/reindex class indices, that requires a careful migration and
should be done only after backups.
"""
import os
import csv
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = REPO_ROOT / 'dataset'
LABELS_CSV = DATASET_ROOT / 'labels.csv'
FEATURE_ROOT = DATASET_ROOT / 'features'


def read_labels():
    if not LABELS_CSV.exists():
        print(f"Labels file not found: {LABELS_CSV}")
        return []
    with open(LABELS_CSV, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def find_gaps(class_indices):
    if not class_indices:
        return []
    mn = min(class_indices)
    mx = max(class_indices)
    full = set(range(mn, mx + 1))
    missing = sorted(list(full - set(class_indices)))
    return missing


def main(args):
    labels = read_labels()
    if not labels:
        print('No labels found.')
        return

    class_idxs = []
    missing_folders = []

    for r in labels:
        try:
            idx = int(r.get('class_idx', 0))
        except Exception:
            continue
        class_idxs.append(idx)
        folder = r.get('folder_name') or ''
        folder_path = FEATURE_ROOT / folder
        if not folder_path.exists():
            missing_folders.append((idx, folder, folder_path))

    class_idxs_sorted = sorted(class_idxs)
    gaps = find_gaps(class_idxs_sorted)

    print('Labels summary:')
    print(f'  total labels: {len(class_idxs_sorted)}')
    print(f'  min class_idx: {class_idxs_sorted[0]}')
    print(f'  max class_idx: {class_idxs_sorted[-1]}')
    print(f'  gaps in sequence: {gaps if gaps else "none"}')
    print('')

    if missing_folders:
        print('Missing folders:')
        for idx, folder, path in missing_folders:
            print(f'  class_idx={idx} folder={folder} path={path}')
    else:
        print('No missing feature folders detected.')

    if args.create_folders and missing_folders:
        for idx, folder, path in missing_folders:
            try:
                path.mkdir(parents=True, exist_ok=True)
                print(f'Created: {path}')
            except Exception as e:
                print(f'Failed to create {path}: {e}')

    if args.show_rows:
        print('\nFull labels rows:')
        for r in labels:
            print(r)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', action='store_true', help='Print report about labels and folders')
    p.add_argument('--create-folders', dest='create_folders', action='store_true', help='Create missing feature folders')
    p.add_argument('--show-rows', action='store_true', help='Dump labels.csv rows')
    args = p.parse_args()

    if not (args.report or args.create_folders or args.show_rows):
        p.print_help()
    else:
        main(args)
