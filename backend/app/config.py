import os
import sys
from typing import List
from pydantic import BaseSettings, validator
from pathlib import Path

def _default_dataset_root() -> Path:
    """Resolve dataset root to repo-level ../dataset by default.
    This avoids writing inside backend/ and works cross-platform.
    """
    # app/config.py -> backend/app -> backend -> repo_root
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "dataset"


def _default_model_root() -> Path:
    """Resolve model root to repo-level ../models/active by default."""
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "models" / "active"


class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/signdb")
    broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
    storage_path: str = os.getenv("STORAGE_PATH", "/app/storage")
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "sign-dataset")

    # Dataset publication/versioning (used for remote object keys)
    dataset_version: str = os.getenv("DATASET_VERSION", "v0")
    dataset_upload_remove_local: bool = bool(int(os.getenv("DATASET_UPLOAD_REMOVE_LOCAL", 0)))

    # Processing constants (align live-capture and video)
    feature_dim: int = int(os.getenv("FEATURE_DIM", 126))
    seq_len: int = int(os.getenv("SEQ_LEN", 60))
    stride: int = int(os.getenv("STRIDE", 2))  # Default spec stride
    fps_target: int = int(os.getenv("FPS_TARGET", 30))
    augment_per_seq: int = int(os.getenv("AUG_PER_SEQ", 8))
    # Video-only augmentation multiplier (defaults to AUG_PER_SEQ)
    video_augment_per_seq: int = int(os.getenv("VIDEO_AUG_PER_SEQ", os.getenv("AUG_PER_SEQ", 8)))
    # Best-effort minimum number of samples saved per video upload (subject to remaining MAX_SAMPLES_PER_CLASS quota).
    # Set 0 to disable.
    video_min_samples_per_video: int = int(os.getenv("VIDEO_MIN_SAMPLES_PER_VIDEO", 10))
    resize_width: int = int(os.getenv("RESIZE_W", 640))
    resize_height: int = int(os.getenv("RESIZE_H", 480))

    # Input preprocessing
    # If the incoming frames are mirrored (common in front-camera previews), flip them back.
    mirror_input: bool = bool(int(os.getenv("MIRROR_INPUT", 0)))
    # Normalize keypoints to reduce translation/scale variance.
    normalize_keypoints: bool = bool(int(os.getenv("NORMALIZE_KEYPOINTS", 0)))
    # Live-capture processing flags
    enable_live_aug: bool = bool(int(os.getenv("ENABLE_LIVE_AUG", 1)))
    enable_live_smoothing: bool = bool(int(os.getenv("ENABLE_LIVE_SMOOTHING", 0)))
    live_completeness_threshold: float = float(os.getenv("LIVE_COMPLETENESS", 0.5))

    # Mirror / handedness invariance
    # Canonicalize per-frame feature vectors so mirrored input (or using left vs right hand)
    # maps to a consistent representation.
    canonicalize_hands: bool = bool(int(os.getenv("CANONICALIZE_HANDS", 1)))
    # If enabled, choose a canonical orientation between (vec) and (mirrored+swapped vec).
    canonicalize_mirror: bool = bool(int(os.getenv("CANONICALIZE_MIRROR", 1)))
    # Carry-forward missing hand frames vs zero-fill
    carry_forward_missing: bool = bool(int(os.getenv("CARRY_FORWARD_MISSING", 1)))
    # Debug logging toggle
    debug_logging: bool = bool(int(os.getenv("DEBUG_LOGGING", 0)))

    # Dataset root: prefer DATASET_ROOT; fallback to repo-level ../dataset
    dataset_root: Path = Path(os.getenv("DATASET_ROOT") or _default_dataset_root())

    # Model root: prefer MODEL_ROOT; fallback to repo-level ../models/active
    model_root: Path = Path(os.getenv("MODEL_ROOT") or _default_model_root())

    # Inference backend selection
    # - tensorflow: use models/active/model_manifest.json (existing behavior)
    # - pytorch: use MODEL_ARTIFACT_PATH (.pth/.pth.tar)
    inference_backend: str = os.getenv("INFERENCE_BACKEND", os.getenv("MODEL_BACKEND", "tensorflow"))
    inference_model_id: str = os.getenv("INFERENCE_MODEL_ID", "active")
    inference_preprocess_version: str = os.getenv("INFERENCE_PREPROCESS_VERSION", "unknown")

    # PyTorch inference artifact inputs
    model_artifact_path: str = os.getenv("MODEL_ARTIFACT_PATH", "")
    model_label_map_path: str = os.getenv("MODEL_LABEL_MAP_PATH", "")
    torch_hidden_size: int = int(os.getenv("TORCH_HIDDEN", 256))
    torch_num_layers: int = int(os.getenv("TORCH_LAYERS", 2))
    torch_temperature: float = float(os.getenv("TORCH_TEMP", 1.4))

    # Inference stability gating (helps avoid confident predictions on idle/noisy input)
    infer_min_energy: float = float(os.getenv("INFER_MIN_ENERGY", 0.00010))
    infer_min_motion: float = float(os.getenv("INFER_MIN_MOTION", 0.00050))

    # Video processing knobs
    # Threshold for window completeness when processing videos (0..1)
    video_completeness_threshold: float = float(os.getenv("VIDEO_COMPLETENESS", 0.8))

    # Video trimming / activity gating
    # Skip leading frames until at least one hand is detected (reduces "idle" prefix).
    video_skip_leading_no_hand: bool = bool(int(os.getenv("VIDEO_SKIP_LEADING_NO_HAND", 1)))
    # Reject windows with low motion even if hands are present.
    # 0 disables this filter. Typical tuning depends on normalization and camera.
    video_activity_threshold: float = float(os.getenv("VIDEO_ACTIVITY_THRESHOLD", 0.0))
    # Comma-separated list of speed variants (e.g., "1.0,1.2,0.8")
    speed_variants_raw: str = os.getenv("SPEED_VARIANTS", "1.0,1.2,0.8")
    # Maximum number of saved samples per class (global cap across the dataset)
    max_samples_per_class: int = int(os.getenv("MAX_SAMPLES_PER_CLASS", 80))
    # Parsed list of speed variants; populated in __init__
    speed_variants: List[float] = [1.0, 1.2, 0.8]

    @validator("speed_variants", pre=True, always=True)
    def _parse_speed_variants(cls, v, values):
        raw = values.get("speed_variants_raw")
        if raw:
            parts = [p.strip() for p in raw.split(",") if p.strip()]
            try:
                return [float(p) for p in parts] if parts else [1.0]
            except Exception:
                return [1.0]
        return v or [1.0]

    def __init__(self, **values):
        super().__init__(**values)
        # If running on POSIX (e.g., Docker/Linux) and DATASET_ROOT was a Windows path,
        # avoid creating a literal 'D:' folder; fallback to default.
        dr_str = str(self.dataset_root)
        if os.name == "posix" and (":\\" in dr_str or ":/" in dr_str):
            self.dataset_root = _default_dataset_root()

        mr_str = str(self.model_root)
        if os.name == "posix" and (":\\" in mr_str or ":/" in mr_str):
            self.model_root = _default_model_root()

        # speed_variants is computed by validator; no assignment needed here

settings = Settings()
