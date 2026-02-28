from __future__ import annotations

from dataclasses import dataclass


class ModelNotReady(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    preprocess_version: str
    sequence_length: int
    feature_dim: int
