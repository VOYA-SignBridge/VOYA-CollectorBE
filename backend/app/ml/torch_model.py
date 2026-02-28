from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from app.config import settings
from app.ml.spec import ModelNotReady, ModelSpec


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # common when saving from DataParallel
    if not state_dict:
        return state_dict
    if all(isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


class TorchCheckpointModel:
    """Loads a PyTorch checkpoint (.pth/.pth.tar) for inference.

    - Lazy-imports torch so the backend can still start without it.
    - Keeps the public API compatible with the existing TF model wrapper.
    - Accepts incoming features with (T, settings.feature_dim) and pads/truncates
      to the checkpoint's input dim if needed.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded: bool = False
        self._load_error: Optional[str] = None
        self._spec: Optional[ModelSpec] = None
        self._model: Any = None
        self._label_map: Optional[Dict[str, Any]] = None
        self._checkpoint_input_dim: Optional[int] = None
        self._pooling: Optional[str] = None
        self._norm_mean: Optional[np.ndarray] = None
        self._norm_std: Optional[np.ndarray] = None

    def _read_json(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _resolve_path(self, raw: str) -> Path:
        p = Path(raw)
        if p.is_absolute():
            return p
        return (settings.model_root / p).resolve()

    def _resolve_artifact(self) -> Path:
        raw = settings.model_artifact_path
        if not raw:
            raise ModelNotReady(
                "Missing MODEL_ARTIFACT_PATH for PyTorch inference. "
                "Example: MODEL_ARTIFACT_PATH=/models/test_run/model_best.pth.tar"
            )
        artifact = self._resolve_path(raw)
        if not artifact.exists():
            raise ModelNotReady(f"Model artifact not found: {artifact}")
        return artifact

    def _resolve_label_map(self, artifact: Path) -> Optional[Path]:
        if settings.model_label_map_path:
            p = self._resolve_path(settings.model_label_map_path)
            return p if p.exists() else None

        # Prefer alongside checkpoint, else fall back to model_root
        cand1 = artifact.parent / "label_map.json"
        if cand1.exists():
            return cand1
        cand2 = settings.model_root / "label_map.json"
        if cand2.exists():
            return cand2
        return None

    def _infer_shapes_from_state_dict(
        self, state_dict: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """Return (checkpoint_input_dim, num_classes, hidden_size, num_layers) if we can infer them."""
        input_dim: Optional[int] = None
        num_classes: Optional[int] = None
        hidden_size: Optional[int] = None
        num_layers: Optional[int] = None

        # Infer GRU input dim + hidden size
        w_ih = state_dict.get("rnn.weight_ih_l0")
        w_hh = state_dict.get("rnn.weight_hh_l0")
        if hasattr(w_ih, "shape"):
            try:
                input_dim = int(w_ih.shape[1])
            except Exception:
                input_dim = None
        if hasattr(w_hh, "shape"):
            try:
                # GRU: (3*hidden, hidden)
                hidden_size = int(w_hh.shape[1])
            except Exception:
                hidden_size = None

        # Infer num_layers from rnn.weight_ih_l{n} keys
        layer_ids = set()
        for k in state_dict.keys():
            if not isinstance(k, str):
                continue
            if k.startswith("rnn.weight_ih_l"):
                tail = k[len("rnn.weight_ih_l") :]
                # tail could be like '0' or '0_reverse'
                num_str = "".join(ch for ch in tail if ch.isdigit())
                if num_str:
                    try:
                        layer_ids.add(int(num_str))
                    except Exception:
                        pass
        if layer_ids:
            num_layers = max(layer_ids) + 1

        # Infer num_classes from the trained head: fc.3.weight (BiGRUModel)
        fc_w = state_dict.get("fc.3.weight")
        if hasattr(fc_w, "shape"):
            try:
                num_classes = int(fc_w.shape[0])
            except Exception:
                num_classes = None

        return input_dim, num_classes, hidden_size, num_layers

    def ensure_loaded(self) -> None:
        with self._lock:
            if self._loaded:
                return

            try:
                try:
                    import torch  # type: ignore
                    import torch.nn as nn  # type: ignore
                except Exception as e:
                    raise ModelNotReady(
                        "PyTorch is not installed in this backend environment. "
                        "Install torch and set INFERENCE_BACKEND=pytorch."
                    ) from e

                artifact = self._resolve_artifact()

                # PyTorch 2.6+ defaults torch.load(weights_only=True), which can fail
                # for checkpoints that include non-tensor metadata (e.g., numpy arrays).
                # Our checkpoints are produced by our own training code, so we explicitly
                # opt into full checkpoint loading.
                try:
                    ckpt = torch.load(str(artifact), map_location="cpu", weights_only=False)
                except TypeError:  # older torch
                    ckpt = torch.load(str(artifact), map_location="cpu")
                if isinstance(ckpt, dict) and "state_dict" in ckpt:
                    state_dict = ckpt.get("state_dict")
                else:
                    state_dict = ckpt

                if not isinstance(state_dict, dict):
                    raise ModelNotReady("Unsupported checkpoint format: expected a state_dict-like dict")

                state_dict = _strip_module_prefix(state_dict)

                inferred_input_dim, inferred_num_classes, inferred_hidden, inferred_layers = self._infer_shapes_from_state_dict(state_dict)
                checkpoint_input_dim = inferred_input_dim
                num_classes = inferred_num_classes

                if num_classes is None:
                    raise ModelNotReady("Unable to infer num_classes from checkpoint state_dict")

                # Match tools/train_baseline.py BiGRUModel checkpoint structure:
                # rnn: GRU(bidirectional), fc: Sequential(Linear->ReLU->Dropout->Linear), forward uses mean pooling.
                input_dim = int(checkpoint_input_dim or settings.feature_dim)
                hidden_size = int(inferred_hidden or settings.torch_hidden_size)
                num_layers = int(inferred_layers or settings.torch_num_layers)

                pooling = "mean"
                if isinstance(ckpt, dict):
                    pooling = str(ckpt.get("pooling") or ckpt.get("args", {}).get("pooling") or "mean")
                if pooling not in {"mean", "last", "attn"}:
                    pooling = "mean"

                class _BiGRUModel(nn.Module):
                    def __init__(self, in_dim: int, hid: int, layers: int, classes: int, pooling: str):
                        super().__init__()
                        self.pooling = pooling
                        self.rnn = nn.GRU(
                            in_dim,
                            hid,
                            num_layers=layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.0,
                        )

                        enc_dim = hid * 2
                        if pooling == "attn":
                            self.attn = nn.Sequential(
                                nn.Linear(enc_dim, 128),
                                nn.Tanh(),
                                nn.Linear(128, 1),
                            )
                        else:
                            self.attn = None

                        self.fc = nn.Sequential(
                            nn.Linear(hid * 2, 128),
                            nn.ReLU(),
                            nn.Dropout(0.0),
                            nn.Linear(128, classes),
                        )

                    def forward(self, x):
                        out, _ = self.rnn(x)
                        if self.pooling == "mean":
                            pooled = out.mean(dim=1)
                        elif self.pooling == "attn" and self.attn is not None:
                            # validity heuristic: frame has any non-zero feature
                            valid = (x.abs().sum(dim=-1) > 1e-8)
                            scores = self.attn(out).squeeze(-1)
                            scores = scores.masked_fill(~valid, -1e9)
                            w = torch.softmax(scores, dim=1).unsqueeze(-1)
                            pooled = (out * w).sum(dim=1)
                        else:
                            valid = (x.abs().sum(dim=-1) > 1e-8)
                            idx = valid.long().sum(dim=1) - 1
                            idx = idx.clamp(min=0)
                            pooled = out[torch.arange(out.shape[0], device=out.device), idx]
                        return self.fc(pooled)

                model = _BiGRUModel(input_dim, hidden_size, num_layers, int(num_classes), pooling=pooling)

                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                # If nothing matches at all, surface a clearer error
                if len(missing) >= len(model.state_dict()):
                    raise ModelNotReady(
                        "Checkpoint architecture mismatch. "
                        f"missing={len(missing)} unexpected={len(unexpected)}"
                    )

                model.eval()

                # Optional normalization stats from checkpoint
                norm_mean = None
                norm_std = None
                if isinstance(ckpt, dict):
                    nm = ckpt.get("norm_mean")
                    ns = ckpt.get("norm_std")
                    if nm is not None and ns is not None:
                        try:
                            norm_mean = np.asarray(nm, dtype=np.float32)
                            norm_std = np.asarray(ns, dtype=np.float32)
                        except Exception:
                            norm_mean = None
                            norm_std = None

                label_map = None
                label_map_path = self._resolve_label_map(artifact)
                if label_map_path:
                    try:
                        label_map = self._read_json(label_map_path)
                    except Exception:
                        label_map = None

                # Best-effort metadata
                model_id = None
                preprocess_version = None
                if isinstance(ckpt, dict):
                    model_id = ckpt.get("model_id") or ckpt.get("run_name")
                    preprocess_version = ckpt.get("preprocess_version")

                self._checkpoint_input_dim = int(input_dim)
                self._model = model
                self._label_map = label_map
                self._pooling = pooling
                self._norm_mean = norm_mean
                self._norm_std = norm_std
                self._spec = ModelSpec(
                    model_id=str(model_id or settings.inference_model_id or artifact.stem),
                    preprocess_version=str(preprocess_version or settings.inference_preprocess_version or "unknown"),
                    sequence_length=int(settings.seq_len),
                    feature_dim=int(settings.feature_dim),
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
                "artifact_path": settings.model_artifact_path,
                "backend": "pytorch",
            }
        assert self._spec is not None
        return {
            "loaded": True,
            "backend": "pytorch",
            "model_id": self._spec.model_id,
            "preprocess_version": self._spec.preprocess_version,
            "sequence_length": self._spec.sequence_length,
            "feature_dim": self._spec.feature_dim,
            "checkpoint_input_dim": self._checkpoint_input_dim,
            "has_label_map": bool(self._label_map),
            "pooling": self._pooling,
            "has_norm": bool(self._norm_mean is not None and self._norm_std is not None),
            "model_root": str(settings.model_root),
            "artifact_path": settings.model_artifact_path,
        }

    def spec(self) -> ModelSpec:
        if not self._loaded or not self._spec:
            raise ModelNotReady("Model not loaded")
        return self._spec

    def _pad_or_truncate_dim(self, x: np.ndarray, target_dim: int) -> np.ndarray:
        if x.shape[-1] == target_dim:
            return x
        if x.shape[-1] > target_dim:
            return x[..., :target_dim]
        pad = target_dim - x.shape[-1]
        return np.pad(x, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        if not self._loaded:
            self.ensure_loaded()
        assert self._model is not None
        assert self._checkpoint_input_dim is not None

        import torch  # type: ignore

        x = np.asarray(features, dtype=np.float32)
        if x.ndim == 3 and x.shape[0] == 1:
            x = x[0]
        if x.ndim != 2:
            raise ValueError("Expected features with shape (T,D) or (1,T,D)")

        x = self._pad_or_truncate_dim(x, int(self._checkpoint_input_dim))

        if self._norm_mean is not None and self._norm_std is not None:
            try:
                m = self._norm_mean
                s = self._norm_std
                # apply on overlap dims
                d = min(x.shape[-1], int(m.shape[0]), int(s.shape[0]))
                x[..., :d] = (x[..., :d] - m[:d]) / s[:d]
            except Exception:
                pass

        xt = torch.from_numpy(x).unsqueeze(0).float()  # (1,T,D)
        with torch.no_grad():
            logits = self._model(xt)
            temp = float(settings.torch_temperature) if getattr(settings, "torch_temperature", 0) else 1.0
            if not np.isfinite(temp) or temp <= 0:
                temp = 1.0
            proba = torch.softmax(logits / temp, dim=-1)
        y = proba.detach().cpu().numpy()
        if y.ndim == 2 and y.shape[0] == 1:
            return y[0]
        raise ValueError(f"Unexpected model output shape: {y.shape}")

    def label_for_index(self, idx: int) -> Optional[Dict[str, Any]]:
        if not self._label_map:
            return None
        return self._label_map.get(str(idx))


TORCH_MODEL = TorchCheckpointModel()
