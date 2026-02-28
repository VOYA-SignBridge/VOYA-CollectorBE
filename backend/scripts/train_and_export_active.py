from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


def _read_labels_csv(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.exists():
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                class_idx = int(row.get("class_idx") or "")
            except Exception:
                continue
            out[class_idx] = {
                "class_uid": row.get("class_uid"),
                "class_idx": class_idx,
                "slug": row.get("slug"),
                "label_original": row.get("label_original"),
                "language": row.get("language"),
                "dialect": row.get("dialect"),
            }
    return out


def _load_memmap(memmap_dir: Path) -> Tuple[np.memmap, np.memmap, Dict[str, Any]]:
    meta_path = memmap_dir / "meta.json"
    if not meta_path.exists():
        raise RuntimeError(f"Missing {meta_path}. Run /api/dataset/export first.")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    shape = meta.get("shape")
    if not (isinstance(shape, list) and len(shape) == 3):
        raise RuntimeError(f"Invalid meta.json shape: {shape}")

    N, T, D = int(shape[0]), int(shape[1]), int(shape[2])

    x_path = Path(meta.get("memmap_path") or (memmap_dir / "features.dat"))
    y_path = Path(meta.get("labels_path") or (memmap_dir / "labels.dat"))

    if not x_path.exists():
        raise RuntimeError(f"Missing features memmap: {x_path}")
    if not y_path.exists():
        raise RuntimeError(
            f"Missing labels memmap: {y_path}. Re-run exporter after updating backend/app/processing/utils.py"
        )

    X = np.memmap(str(x_path), dtype=np.float32, mode="r", shape=(N, T, D))
    y = np.memmap(str(y_path), dtype=np.int32, mode="r", shape=(N,))
    return X, y, {"N": N, "T": T, "D": D, **meta}


def main() -> None:
    dataset_root = Path(os.environ.get("DATASET_ROOT", "/dataset"))
    memmap_dir = dataset_root / "processed" / "memmap"

    model_root = Path(os.environ.get("MODEL_ROOT", "/models/active"))
    model_root.mkdir(parents=True, exist_ok=True)

    epochs = _env_int("EPOCHS", 3)
    batch_size = _env_int("BATCH_SIZE", 32)
    max_samples = _env_int("MAX_SAMPLES", 0)

    X, y_raw, meta = _load_memmap(memmap_dir)
    N, T, D = int(meta["N"]), int(meta["T"]), int(meta["D"])

    valid_mask = y_raw[:] >= 0
    valid_idx = np.where(valid_mask)[0]
    if valid_idx.size == 0:
        raise RuntimeError(
            "No labeled samples found (all class_idx are missing). Ensure your .npz meta has class_idx."
        )

    if max_samples and valid_idx.size > max_samples:
        valid_idx = valid_idx[:max_samples]

    y_vals = y_raw[valid_idx]
    unique_classes = sorted({int(v) for v in y_vals.tolist()})
    class_to_out = {ci: i for i, ci in enumerate(unique_classes)}
    y = np.asarray([class_to_out[int(v)] for v in y_vals.tolist()], dtype=np.int32)

    # Materialize a contiguous array for training (Keras doesn't train well on memmap slices)
    X_train_full = np.asarray(X[valid_idx, :, :], dtype=np.float32)

    # Train/val split
    rng = np.random.default_rng(1337)
    perm = rng.permutation(X_train_full.shape[0])
    split = int(0.8 * perm.size)
    train_idx, val_idx = perm[:split], perm[split:]

    X_tr, y_tr = X_train_full[train_idx], y[train_idx]
    X_va, y_va = X_train_full[val_idx], y[val_idx]

    num_classes = len(unique_classes)

    import tensorflow as tf  # type: ignore

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(T, D)),
            tf.keras.layers.Masking(mask_value=0.0),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_va, y_va) if X_va.shape[0] else None,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    artifact_path = model_root / "model.h5"
    model.save(str(artifact_path))

    # Build label_map.json for inference responses
    labels_csv = dataset_root / "labels.csv"
    label_rows = _read_labels_csv(labels_csv)

    label_map: Dict[str, Any] = {}
    for out_idx, original_ci in enumerate(unique_classes):
        row = label_rows.get(int(original_ci)) or {}
        label_map[str(out_idx)] = {
            "class_uid": row.get("class_uid"),
            "slug": row.get("slug"),
            "label_original": row.get("label_original") or str(original_ci),
            "original_class_idx": int(original_ci),
        }

    (model_root / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model_id = os.environ.get("MODEL_ID") or "trained-lstm-v0"
    preprocess_version = os.environ.get("PREPROCESS_VERSION") or "mp_hands_v1"

    manifest = {
        "model_id": model_id,
        "preprocess_version": preprocess_version,
        "input_spec": {"sequence_length": int(T), "feature_dim": int(D)},
        "artifact_path": artifact_path.name,
        "num_classes": int(num_classes),
        "notes": {
            "trained_on_total": int(valid_idx.size),
            "unique_original_class_idx": unique_classes,
        },
    }
    (model_root / "model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "status": "ok",
                "memmap_dir": str(memmap_dir),
                "trained_samples": int(valid_idx.size),
                "T": int(T),
                "D": int(D),
                "num_classes": int(num_classes),
                "model_root": str(model_root),
                "artifact": str(artifact_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
