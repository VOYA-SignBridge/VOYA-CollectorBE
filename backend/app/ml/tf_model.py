from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from app.config import settings
from app.ml.spec import ModelNotReady, ModelSpec


class TfSavedModel:
    """Loads a TensorFlow model for inference (SavedModel dir or .h5).

    This module is intentionally lazy-importing tensorflow so the backend can still start
    in environments where TF isn't installed (endpoints will return a clear error).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: bool = False
        self._load_error: Optional[str] = None
        self._spec: Optional[ModelSpec] = None
        self._manifest: Optional[Dict[str, Any]] = None
        self._model: Any = None
        self._label_map: Optional[Dict[str, Any]] = None

    def _read_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _resolve_artifact_path(self, artifact_path: str) -> Path:
        p = Path(artifact_path)
        if p.is_absolute():
            return p
        return (settings.model_root / p).resolve()

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return
            try:
                import tensorflow as tf  # type: ignore

                manifest_path = settings.model_root / "model_manifest.json"
                if not manifest_path.exists():
                    raise ModelNotReady(
                        f"Missing {manifest_path}. Put your exported model manifest there."
                    )

                manifest = self._read_json(manifest_path)
                model_id = str(manifest.get("model_id") or "")
                preprocess_version = str(manifest.get("preprocess_version") or "")

                input_spec = manifest.get("input_spec") or {}
                seq_len = int(input_spec.get("sequence_length") or input_spec.get("seq_len") or settings.seq_len)
                feat_dim = int(input_spec.get("feature_dim") or input_spec.get("feature_dim") or settings.feature_dim)

                artifact_path = manifest.get("artifact_path") or manifest.get("artifact")
                if not artifact_path:
                    raise ModelNotReady("Manifest missing 'artifact_path' (SavedModel dir or .h5 path).")

                artifact = self._resolve_artifact_path(str(artifact_path))
                if not artifact.exists():
                    raise ModelNotReady(f"Model artifact not found: {artifact}")

                # Keras can load both SavedModel directories and .h5
                model = tf.keras.models.load_model(str(artifact))

                label_map_path = settings.model_root / "label_map.json"
                label_map = None
                if label_map_path.exists():
                    try:
                        label_map = self._read_json(label_map_path)
                    except Exception:
                        label_map = None

                self._model = model
                self._manifest = manifest
                self._label_map = label_map
                self._spec = ModelSpec(
                    model_id=model_id or "active",
                    preprocess_version=preprocess_version or "unknown",
                    sequence_length=seq_len,
                    feature_dim=feat_dim,
                )
                self._loaded = True
            except Exception as e:
                self._load_error = str(e)
                raise

    def status(self) -> Dict[str, Any]:
        if not self._loaded:
            return {
                "loaded": False,
                "error": self._load_error,
                "model_root": str(settings.model_root),
            }
        assert self._spec is not None
        return {
            "loaded": True,
            "model_id": self._spec.model_id,
            "preprocess_version": self._spec.preprocess_version,
            "sequence_length": self._spec.sequence_length,
            "feature_dim": self._spec.feature_dim,
            "has_label_map": bool(self._label_map),
            "model_root": str(settings.model_root),
        }

    def spec(self) -> ModelSpec:
        if not self._loaded or not self._spec:
            raise ModelNotReady("Model not loaded")
        return self._spec

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """features: (T,D) or (1,T,D). Returns (C,) for single sample."""
        if not self._loaded:
            self.ensure_loaded()
        assert self._model is not None

        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 2:
            x = x.reshape(1, x.shape[0], x.shape[1])
        if x.ndim != 3:
            raise ValueError("Expected features with shape (T,D) or (1,T,D)")

        y = self._model.predict(x, verbose=0)
        y = np.asarray(y)
        if y.ndim == 2 and y.shape[0] == 1:
            return y[0]
        raise ValueError(f"Unexpected model output shape: {y.shape}")

    def label_for_index(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self._label_map:
            return None
        return self._label_map.get(str(idx))


TF_MODEL = TfSavedModel()
