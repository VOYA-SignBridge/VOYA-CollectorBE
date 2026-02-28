"""migrate_flat_to_hierarchy
Migrate existing flat dataset stored under dataset/features/class_XXXX_slug
into the new multilingual/multi-dialect hierarchy using dataset_manager and
dataset_samples utilities.

Default behavior:
- Registers each old label as a language-common class under the provided language (default 'vn').
- Re-saves each .npz sample into the new directory with preserved metadata where possible.
- Does not delete or modify old files (idempotent only if you later remove old dirs to avoid duplication).

Run examples (inside backend working directory):
  python -m app.migrations.migrate_flat_to_hierarchy --language vn --language-common

"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import numpy as np

from app.processing import storage_utils as legacy
from app.dataset_manager import ensure_structure, get_or_register_class
from app.dataset_samples import save_sequence_npz


def migrate(language: str = "vn", as_language_common: bool = True, limit: int | None = None, dry_run: bool = False) -> dict:
    ensure_structure()

    labels = legacy.read_csv(legacy.LABELS_CSV)
    if not labels:
        return {"migrated": 0, "skipped": 0, "message": "No legacy labels.csv found or empty."}

    migrated = 0
    skipped = 0
    for i, row in enumerate(labels):
        if limit is not None and migrated >= limit:
            break
        label = row.get("label_original") or row.get("slug") or row.get("folder_name")
        folder_name = row.get("folder_name")
        if not label or not folder_name:
            skipped += 1
            continue

        src_dir = Path(legacy.FEATURE_ROOT) / folder_name
        if not src_dir.exists():
            skipped += 1
            continue

        class_meta = get_or_register_class(
            label_original=label,
            language=language,
            dialect="common" if as_language_common else "",
            is_common_language=as_language_common,
        )

        for fname in os.listdir(src_dir):
            if not fname.endswith(".npz"):
                continue
            fpath = src_dir / fname
            try:
                with np.load(fpath, allow_pickle=True) as npz:
                    sequence = npz.get("sequence")
                    meta = npz.get("meta", {})
                    if isinstance(meta, np.ndarray) and meta.dtype == object and meta.shape == ():
                        # unpack object scalar to dict
                        meta = dict(meta.item())
                    elif meta is None:
                        meta = {}
                    # normalize meta fields
                    normalized = {
                        "user": meta.get("user", ""),
                        "session_id": meta.get("session_id", ""),
                        "fps_original": meta.get("fps"),
                        "fps_processed": meta.get("fps"),
                        "completeness": meta.get("completeness"),
                        "created_at": meta.get("created_at"),
                        "legacy_class_idx": meta.get("class_idx"),
                        "legacy_folder": meta.get("folder_name") or folder_name,
                    }
                    augment_id = int(meta.get("aug_id", 0)) if meta else 0
                    source_type = meta.get("source", "unknown") if meta else "unknown"

                if sequence is None:
                    skipped += 1
                    continue

                if dry_run:
                    migrated += 1
                    continue

                # Ensure float32 2D
                sequence = np.asarray(sequence, dtype=np.float32)
                if sequence.ndim != 2:
                    # attempt to flatten per frame if needed
                    T = sequence.shape[0]
                    sequence = sequence.reshape(T, -1)

                save_sequence_npz(class_meta, sequence, meta=normalized, augment_id=augment_id, source_type=source_type)
                migrated += 1
            except Exception as e:
                print(f"[MIGRATE][WARN] failed on {fpath}: {e}")
                skipped += 1

    return {"migrated": migrated, "skipped": skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="vn", help="Target language code (default vn)")
    parser.add_argument("--language-common", action="store_true", help="Register as language common (dialect=common)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files migrated")
    parser.add_argument("--dry-run", action="store_true", help="Do not write; only count")
    parser.add_argument("--source-root", default=None, help="Override legacy features root (path to dataset/features)")
    parser.add_argument("--legacy-labels", default=None, help="Override legacy labels.csv path (defaults to <dataset>/labels.csv inferred from source-root)")
    args = parser.parse_args()
    # Optionally override legacy paths if provided
    if args.source_root:
        try:
            from app.processing import storage_utils as legacy
            src = Path(args.source_root).expanduser().resolve()
            legacy.FEATURE_ROOT = str(src)
            if args.legacy_labels:
                legacy.LABELS_CSV = str(Path(args.legacy_labels).expanduser().resolve())
            else:
                # infer dataset root as parent of features folder
                dataset_dir = src.parent
                legacy.LABELS_CSV = str(dataset_dir / "labels.csv")
        except Exception as e:
            print(f"[MIGRATE][WARN] failed to set legacy paths: {e}")
    res = migrate(language=args.language, as_language_common=args.language_common, limit=args.limit, dry_run=args.dry_run)
    print(res)


if __name__ == "__main__":
    main()
