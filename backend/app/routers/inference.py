from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.config import settings
from app.inference_loader import load_inference_classes, build_class_map
from app.ml.spec import ModelNotReady
from app.ml.tf_model import TF_MODEL
from app.ml.torch_model import TORCH_MODEL


# Legacy prefix (kept for backward compatibility)
router = APIRouter(prefix="/inference", tags=["inference"])

# New API prefix (matches existing /api/dataset/export style)
api_router = APIRouter(prefix="/api/inference", tags=["inference"])


def _get_engine():
    backend = (settings.inference_backend or "").strip().lower()
    if backend in {"pytorch", "torch", "pt"}:
        return TORCH_MODEL

    # Auto-select torch if user explicitly points to a torch artifact.
    artifact = (settings.model_artifact_path or "").strip().lower()
    if artifact.endswith((".pth", ".pth.tar")):
        return TORCH_MODEL

    return TF_MODEL


@router.get("/classes")
@api_router.get("/classes")
def inference_classes(language: str = Query("vn"), dialect: str = Query("")):
    classes = load_inference_classes(language, dialect)
    return {"language": language, "dialect": dialect, "count": len(classes), "classes": build_class_map(classes)}


@api_router.get("/model")
def get_active_model() -> Dict[str, Any]:
    engine = _get_engine()
    try:
        engine.ensure_loaded()
    except ModelNotReady as e:
        # Model isn't configured yet
        return {"status": "not_ready", **engine.status(), "message": str(e)}
    except Exception as e:
        return {"status": "error", **engine.status(), "message": str(e)}
    return {"status": "ok", **engine.status()}


class SequencePayload(BaseModel):
    sequence_length: int = Field(..., ge=1)
    feature_dim: int = Field(..., ge=1)
    features: List[List[float]]


class PredictFeaturesRequest(BaseModel):
    model_id: Optional[str] = None
    preprocess_version: Optional[str] = None
    top_k: int = Field(3, ge=1, le=10)
    language: Optional[str] = None
    dialect: Optional[str] = None
    sequence: SequencePayload


class PredictItem(BaseModel):
    class_idx: int
    confidence: float
    class_uid: Optional[str] = None
    slug: Optional[str] = None
    label_original: Optional[str] = None


class PredictFeaturesResponse(BaseModel):
    model_id: str
    preprocess_version: str
    prediction: PredictItem
    top_k: List[PredictItem]
    timing_ms: Dict[str, int]
    skipped: bool = False
    skip_reason: Optional[str] = None
    input_stats: Optional[Dict[str, float]] = None


@api_router.post("/predict/features", response_model=PredictFeaturesResponse)
def predict_features(req: PredictFeaturesRequest):
    start = time.time()

    engine = _get_engine()

    try:
        engine.ensure_loaded()
        spec = engine.spec()
    except ModelNotReady as e:
        raise HTTPException(status_code=503, detail=f"Model not ready: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Version checks (soft-fail if client didn't send; strict if sent)
    if req.model_id and req.model_id != spec.model_id:
        raise HTTPException(status_code=400, detail=f"model_id mismatch: {req.model_id} != {spec.model_id}")
    if req.preprocess_version and req.preprocess_version != spec.preprocess_version:
        raise HTTPException(
            status_code=400,
            detail=f"preprocess_version mismatch: {req.preprocess_version} != {spec.preprocess_version}",
        )

    T = req.sequence.sequence_length
    D = req.sequence.feature_dim
    if T != spec.sequence_length:
        raise HTTPException(status_code=400, detail=f"sequence_length mismatch: {T} != {spec.sequence_length}")
    if D != spec.feature_dim:
        raise HTTPException(status_code=400, detail=f"feature_dim mismatch: {D} != {spec.feature_dim}")

    # Validate payload shape quickly
    if len(req.sequence.features) != T:
        raise HTTPException(status_code=400, detail=f"features length mismatch: got {len(req.sequence.features)} frames, expected {T}")
    for i, row in enumerate(req.sequence.features[:3]):
        if len(row) != D:
            raise HTTPException(status_code=400, detail=f"feature row {i} dim mismatch: got {len(row)}, expected {D}")

    validate_ms = int((time.time() - start) * 1000)

    x = np.asarray(req.sequence.features, dtype=np.float32)
    # Backend-side safety: avoid NaN/Inf poisoning the model
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Stability gating: if the signal is too weak (idle / mostly zeros), avoid confident predictions.
    energy = float(np.mean(np.abs(x)))
    motion = 0.0
    if x.shape[0] > 1:
        motion = float(np.mean(np.abs(np.diff(x, axis=0))))

    min_energy = float(getattr(settings, "infer_min_energy", 0.0) or 0.0)
    min_motion = float(getattr(settings, "infer_min_motion", 0.0) or 0.0)

    infer_start = time.time()
    skipped = False
    skip_reason: Optional[str] = None

    if (min_energy and energy < min_energy) or (min_motion and motion < min_motion):
        skipped = True
        reasons = []
        if min_energy and energy < min_energy:
            reasons.append(f"low_energy({energy:.6f}<{min_energy:.6f})")
        if min_motion and motion < min_motion:
            reasons.append(f"low_motion({motion:.6f}<{min_motion:.6f})")
        skip_reason = ",".join(reasons) if reasons else "low_signal"

        # Return a low-confidence, uniform distribution (frontend won't lock-in).
        # We still need class count; best-effort: run model only if it is cheap/available.
        proba = engine.predict_proba(x)
        C = int(proba.shape[0]) if getattr(proba, "shape", None) is not None else 0
        if C > 0:
            proba = np.full((C,), 1.0 / float(C), dtype=np.float32)
    else:
        proba = engine.predict_proba(x)
    infer_ms = int((time.time() - infer_start) * 1000)

    if proba.ndim != 1:
        raise HTTPException(status_code=500, detail=f"Unexpected proba shape: {proba.shape}")

    proba = np.asarray(proba, dtype=np.float32)
    proba = np.nan_to_num(proba, nan=0.0, posinf=0.0, neginf=0.0)
    proba = np.clip(proba, 0.0, 1.0)
    s = float(np.sum(proba))
    if not np.isfinite(s) or s <= 0:
        proba = np.full_like(proba, 1.0 / float(max(1, proba.shape[0])))
    else:
        proba = proba / s

    k = min(req.top_k, int(proba.shape[0]))
    top_idx = np.argsort(-proba)[:k]

    def _item(i: int) -> PredictItem:
        conf = float(proba[i])
        meta = engine.label_for_index(int(i))
        if meta:
            return PredictItem(
                class_idx=int(i),
                confidence=conf,
                class_uid=meta.get("class_uid"),
                slug=meta.get("slug"),
                label_original=meta.get("label_original"),
            )
        return PredictItem(class_idx=int(i), confidence=conf)

    top_items = [_item(int(i)) for i in top_idx]
    pred = top_items[0]

    total_ms = int((time.time() - start) * 1000)
    return PredictFeaturesResponse(
        model_id=spec.model_id,
        preprocess_version=spec.preprocess_version,
        prediction=pred,
        top_k=top_items,
        timing_ms={"validate": validate_ms, "inference": infer_ms, "total": total_ms},
        skipped=skipped,
        skip_reason=skip_reason,
        input_stats={"energy": energy, "motion": motion, "min_energy": min_energy, "min_motion": min_motion},
    )
