"""dataset_samples.py
Unified sample metadata storage for multilingual/multi-dialect dataset.

Fields stored (samples.csv):
 sample_uid,class_uid,slug,label_original,language,dialect,source_type,user_id,session_id,
 fps_original,fps_processed,seq_len,augment_id,completeness,file_path,created_at
"""
from __future__ import annotations

import csv
import os
import uuid
import json
from pathlib import Path
from typing import Dict, Any, List
from filelock import FileLock
from datetime import datetime
from app.config import settings
import logging

DATASET_ROOT = settings.dataset_root
SAMPLES_DIR = DATASET_ROOT / "samples"
SAMPLES_CSV = SAMPLES_DIR / "samples.csv"
# Legacy compatibility: root-level samples CSV expected by older tooling
LEGACY_SAMPLES_CSV = DATASET_ROOT / "samples.csv"

SAMPLE_FIELDS = [
    "sample_uid","class_uid","slug","label_original","language","dialect","source_type",
    "user_id","session_id","fps_original","fps_processed","seq_len","augment_id","completeness",
    "file_path","created_at"
]

def now_str() -> str:
    return datetime.utcnow().isoformat() + "Z"

def _ensure_samples_file():
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    if not SAMPLES_CSV.exists():
        lock = FileLock(str(SAMPLES_CSV)+".lock")
        with lock:
            if not SAMPLES_CSV.exists():
                with open(SAMPLES_CSV, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
                    writer.writeheader()
    # Ensure legacy root-level samples CSV exists
    if not LEGACY_SAMPLES_CSV.exists():
        lock = FileLock(str(LEGACY_SAMPLES_CSV)+".lock")
        with lock:
            if not LEGACY_SAMPLES_CSV.exists():
                with open(LEGACY_SAMPLES_CSV, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        "sample_id","class_idx","folder_name","file","user","session_id","frames","duration","source","dialect","created_at"
                    ])
                    writer.writeheader()

def append_sample_row(row: Dict[str, Any]):
    _ensure_samples_file()
    lock = FileLock(str(SAMPLES_CSV)+".lock")
    with lock:
        file_exists = SAMPLES_CSV.exists()
        with open(SAMPLES_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SAMPLE_FIELDS)
            if not file_exists or os.path.getsize(SAMPLES_CSV)==0:
                writer.writeheader()
            writer.writerow(row)
            f.flush(); os.fsync(f.fileno())

def list_samples() -> List[Dict[str,str]]:
    _ensure_samples_file()
    lock = FileLock(str(SAMPLES_CSV)+".lock")
    with lock:
        with open(SAMPLES_CSV, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))


def count_samples_for_class(class_uid: str) -> int:
    """Return number of samples for a class_uid based on samples.csv.

    This is the source of truth for enforcing a global per-class cap (e.g. MAX_SAMPLES_PER_CLASS).
    """
    if not class_uid:
        return 0
    _ensure_samples_file()
    lock = FileLock(str(SAMPLES_CSV)+".lock")
    with lock:
        try:
            with open(SAMPLES_CSV, newline="", encoding="utf-8") as f:
                return sum(1 for row in csv.DictReader(f) if row.get("class_uid") == class_uid)
        except FileNotFoundError:
            return 0

def save_sequence_npz(class_meta, sequence, meta: Dict[str, Any], augment_id: int, source_type: str) -> str:
    """Save a (T,D) sequence under the class hierarchy with extended metadata.
    Returns the npz file path.
    """
    class_dir = class_meta.hierarchy_path()
    class_dir.mkdir(parents=True, exist_ok=True)
    sample_uid = uuid.uuid4().hex[:10]
    created_at = (meta or {}).get("created_at") or now_str()
    fname = f"sample_{sample_uid}.npz"
    fpath = class_dir / fname
    sidecar = class_dir / f"sample_{sample_uid}.json"
    # Write npz atomically
    import numpy as np, tempfile
    fd, tmp = tempfile.mkstemp(prefix="npztmp_", suffix=".npz", dir=str(class_dir))
    os.close(fd)
    metadata = {
        "class_uid": class_meta.class_uid,
        "slug": class_meta.slug,
        "label_original": class_meta.label_original,
        "language": class_meta.language,
        "dialect": class_meta.dialect,
        "augment_id": augment_id,
        "created_at": created_at,
        **meta,
    }
    try:
        with open(tmp, "wb") as f:
            np.savez_compressed(f, sequence=sequence.astype("float32"), meta=metadata)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, fpath)
    finally:
        if os.path.exists(tmp):
            try: os.remove(tmp)
            except: pass
    # Optionally upload to MinIO and compute checksum
    storage_url = None
    checksum = None
    try:
        from app.storage.minio_client import upload_file, compute_sha256
        if settings.minio_endpoint and settings.minio_access_key and settings.minio_secret_key:
            checksum = compute_sha256(str(fpath))
            # Build key: features/<language>/<dialect>/<class_uid>/<fname>
            key = f"features/{getattr(settings, 'dataset_version', 'v0')}/{class_meta.language}/{class_meta.dialect}/{class_meta.class_uid}/{fpath.name}"
            storage_url = upload_file(str(fpath), key)
            if storage_url:
                metadata['storage_url'] = storage_url
                metadata['checksum'] = checksum
                # Optionally remove local file if configured
                if getattr(settings, 'dataset_upload_remove_local', False):
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
    except Exception:
        # Best-effort: do not fail the pipeline on upload errors
        storage_url = None
        checksum = None
    # Write sidecar
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Append sample record
    append_sample_row({
        "sample_uid": sample_uid,
        "class_uid": class_meta.class_uid,
        "slug": class_meta.slug,
        "label_original": class_meta.label_original,
        "language": class_meta.language,
        "dialect": class_meta.dialect,
        "source_type": source_type,
        "user_id": meta.get("user",""),
        "session_id": meta.get("session_id",""),
        "fps_original": meta.get("fps_original", meta.get("fps", "")),
        "fps_processed": meta.get("fps_processed", meta.get("fps", "")),
        "seq_len": str(sequence.shape[0]),
        "augment_id": str(augment_id),
        "completeness": str(meta.get("completeness","")),
        "file_path": str(fpath),
        "created_at": created_at,
    })
    # Also persist metadata to Postgres if configured
    try:
        from app.storage.metadata_db import insert_sample
        db_row = {
            "sample_uid": sample_uid,
            "class_uid": class_meta.class_uid,
            "slug": class_meta.slug,
            "label_original": class_meta.label_original,
            "language": class_meta.language,
            "dialect": class_meta.dialect,
            "source_type": source_type,
            "user_id": meta.get("user",""),
            "session_id": meta.get("session_id",""),
            "fps_original": meta.get("fps_original", meta.get("fps", "")),
            "fps_processed": meta.get("fps_processed", meta.get("fps", "")),
            "seq_len": int(sequence.shape[0]),
            "augment_id": int(augment_id),
            "completeness": float(meta.get("completeness") or 0.0),
            "file_path": str(fpath),
            "storage_url": metadata.get("storage_url"),
            "checksum": metadata.get("checksum"),
            "created_at": created_at,
        }
        insert_sample(db_row)
    except Exception as e:
        # Best-effort: do not fail the pipeline if DB is unavailable
        if getattr(settings, "debug_logging", False):
            logging.getLogger(__name__).debug("[DB] insert_sample failed: %s", e)
    # Append legacy-format sample record to root-level CSV
    try:
        lock_legacy = FileLock(str(LEGACY_SAMPLES_CSV)+".lock")
        with lock_legacy:
            file_exists = LEGACY_SAMPLES_CSV.exists()
            with open(LEGACY_SAMPLES_CSV, "a", newline="", encoding="utf-8") as f:
                fields = [
                    "sample_id","class_idx","folder_name","file","user","session_id","frames","duration","source","dialect","created_at"
                ]
                writer = csv.DictWriter(f, fieldnames=fields)
                if not file_exists or os.path.getsize(LEGACY_SAMPLES_CSV)==0:
                    writer.writeheader()
                writer.writerow({
                    "sample_id": sample_uid,
                    "class_idx": str(class_meta.class_idx or ""),
                    "folder_name": class_meta.folder_name(),
                    "file": fpath.name,
                    "user": meta.get("user",""),
                    "session_id": meta.get("session_id",""),
                    "frames": str(sequence.shape[0]),
                    "duration": "",  # optional; leave blank to match legacy examples
                    "source": source_type,
                    "dialect": class_meta.dialect,
                    "created_at": created_at,
                })
                f.flush(); os.fsync(f.fileno())
    except Exception:
        # Best effort; ignore legacy write failure
        pass
    return str(fpath)
