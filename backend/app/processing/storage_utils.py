"""
storage_utils.py
Utility functions for label management, dataset storage, and metadata tracking.
Designed to be production-ready with CSV as source of truth (DB-ready in future).
"""

import os
import csv
import uuid
import unicodedata
import re
import json
import shutil
import tempfile
import io
from datetime import datetime
from filelock import FileLock

# ---- Config paths ----
DATASET_ROOT = "dataset"
FEATURE_ROOT = os.path.join(DATASET_ROOT, "features")
LABELS_CSV = os.path.join(DATASET_ROOT, "labels.csv")
SAMPLES_CSV = os.path.join(DATASET_ROOT, "samples.csv")

# ---- Utils ----
def slugify(text: str, maxlen: int = 20) -> str:
    """Convert text (possibly with diacritics) to safe ASCII slug."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s-]+", "-", text).strip("-")
    if len(text) > maxlen:
        text = text[:maxlen].rstrip("-")
    return text if text else "label"

def now_str() -> str:
    return datetime.utcnow().isoformat() + "Z"

# ---- CSV helpers ----
def read_csv(csv_path):
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def write_csv(csv_path, rows, fieldnames):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    lock = FileLock(csv_path + ".lock")
    with lock:
        # write atomically via temp file then replace
        dirn = os.path.dirname(csv_path) or "."
        fd, tmp_path = tempfile.mkstemp(prefix="csvtmp_", dir=dirn)
        os.close(fd)
        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, csv_path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

def append_csv_row(csv_path, row, fieldnames):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    lock = FileLock(csv_path + ".lock")
    with lock:
        file_exists = os.path.exists(csv_path)
        # Atomic append via opening in a+ and ensure header if new
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())

# ---- Label management ----
def register_label(label_original, notes="", dataset_version="v1"):
    """Register new label or return existing one. Returns (class_idx, folder_name)."""
    rows = read_csv(LABELS_CSV)
    for r in rows:
        if r["label_original"] == label_original:
            return int(r["class_idx"]), r["folder_name"]

    next_idx = max([int(r["class_idx"]) for r in rows], default=0) + 1
    slug = slugify(label_original, maxlen=20)
    folder_name = f"class_{next_idx:04d}_{slug}"
    created_at = now_str()

    new_row = {
        "class_idx": str(next_idx),
        "label_original": label_original,
        "slug": slug,
        "folder_name": folder_name,
        "created_at": created_at,
        "dataset_version": dataset_version,
        "notes": notes,
    }
    # append row atomically
    fieldnames = ["class_idx","label_original","slug","folder_name","created_at","dataset_version","notes"]
    append_csv_row(LABELS_CSV, new_row, fieldnames)

    os.makedirs(os.path.join(FEATURE_ROOT, folder_name), exist_ok=True)
    return next_idx, folder_name

# ---- Sample management ----
def save_sample(sequence_array, class_idx, folder_name, metadata=None):
    """
    Save npz + json metadata in the correct folder.
    Returns file path.
    """
    sample_uuid = uuid.uuid4().hex[:8]
    fname = f"sample_{class_idx:04d}_{sample_uuid}"
    out_dir = os.path.join(FEATURE_ROOT, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, fname + ".npz")
    json_path = os.path.join(out_dir, fname + ".json")

    # Save npz atomically (include meta inside npz too)
    import numpy as np
    metadata = metadata or {}
    metadata.update({
        "class_idx": class_idx,
        "folder_name": folder_name,
        "sample_uuid": sample_uuid,
        "created_at": now_str(),
    })
    fd_npz, tmp_npz = tempfile.mkstemp(prefix="npztmp_", suffix=".npz", dir=out_dir)
    os.close(fd_npz)
    try:
        with open(tmp_npz, "wb") as f:
            np.savez_compressed(f, sequence=sequence_array.astype("float32"), meta=metadata)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_npz, npz_path)
    finally:
        if os.path.exists(tmp_npz):
            try:
                os.remove(tmp_npz)
            except Exception:
                pass

    # Save metadata sidecar atomically
    fd_json, tmp_json = tempfile.mkstemp(prefix="jsontmp_", suffix=".json", dir=out_dir)
    os.close(fd_json)
    try:
        with open(tmp_json, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_json, json_path)
    finally:
        if os.path.exists(tmp_json):
            try:
                os.remove(tmp_json)
            except Exception:
                pass

    # Record in samples.csv
    add_sample_record(fname + ".npz", class_idx, folder_name, metadata)

    return npz_path

def add_sample_record(filename, class_idx, folder_name, metadata):
    new_row = {
        "sample_id": uuid.uuid4().hex[:8],
        "class_idx": str(class_idx),
        "folder_name": folder_name,
        "file": filename,
        "user": metadata.get("user", ""),
        "session_id": metadata.get("session_id", ""),
        "frames": str(metadata.get("frames", "")),
        "duration": str(metadata.get("duration", "")),
        "source": metadata.get("source", ""),
        "dialect": metadata.get("dialect", ""),
        "created_at": metadata.get("created_at", now_str()),
    }
    fieldnames = ["sample_id","class_idx","folder_name","file","user","session_id","frames","duration","source","dialect","created_at"]
    append_csv_row(SAMPLES_CSV, new_row, fieldnames)

# ---- Label merge ----
def merge_labels(src_class_idx, dst_class_idx):
    """
    Merge all samples from src into dst. Update samples.csv and move files.
    """
    label_rows = read_csv(LABELS_CSV)
    samples = read_csv(SAMPLES_CSV)

    # Find folder names
    src_label = next(r for r in label_rows if int(r["class_idx"]) == src_class_idx)
    dst_label = next(r for r in label_rows if int(r["class_idx"]) == dst_class_idx)
    src_folder = os.path.join(FEATURE_ROOT, src_label["folder_name"])
    dst_folder = os.path.join(FEATURE_ROOT, dst_label["folder_name"])

    # Move files
    if os.path.exists(src_folder):
        for fname in os.listdir(src_folder):
            shutil.move(os.path.join(src_folder, fname), os.path.join(dst_folder, fname))

    # Update samples.csv
    for row in samples:
        if int(row["class_idx"]) == src_class_idx:
            row["class_idx"] = str(dst_class_idx)
            row["folder_name"] = dst_label["folder_name"]

    write_csv(SAMPLES_CSV, samples, samples[0].keys())

    # Remove src label from labels.csv
    label_rows = [r for r in label_rows if int(r["class_idx"]) != src_class_idx]
    write_csv(LABELS_CSV, label_rows, label_rows[0].keys())

    # Cleanup
    if os.path.exists(src_folder):
        shutil.rmtree(src_folder, ignore_errors=True)

    return True
