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


class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/signdb")
    broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")
    result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://redis:6379/0")
    storage_path: str = os.getenv("STORAGE_PATH", "/app/storage")
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT")
    minio_access_key: str = os.getenv("MINIO_ACCESS_KEY")
    minio_secret_key: str = os.getenv("MINIO_SECRET_KEY")
    minio_bucket: str = os.getenv("MINIO_BUCKET", "sign-dataset")

    # Processing constants (align live-capture and video)
    feature_dim: int = int(os.getenv("FEATURE_DIM", 126))
    seq_len: int = int(os.getenv("SEQ_LEN", 60))
    stride: int = int(os.getenv("STRIDE", 2))  # Default spec stride
    fps_target: int = int(os.getenv("FPS_TARGET", 30))
    augment_per_seq: int = int(os.getenv("AUG_PER_SEQ", 8))
    resize_width: int = int(os.getenv("RESIZE_W", 640))
    resize_height: int = int(os.getenv("RESIZE_H", 480))
    # Live-capture processing flags
    enable_live_aug: bool = bool(int(os.getenv("ENABLE_LIVE_AUG", 1)))
    enable_live_smoothing: bool = bool(int(os.getenv("ENABLE_LIVE_SMOOTHING", 0)))
    live_completeness_threshold: float = float(os.getenv("LIVE_COMPLETENESS", 0.5))
    # Carry-forward missing hand frames vs zero-fill
    carry_forward_missing: bool = bool(int(os.getenv("CARRY_FORWARD_MISSING", 1)))
    # Debug logging toggle
    debug_logging: bool = bool(int(os.getenv("DEBUG_LOGGING", 0)))

    # Dataset root: prefer DATASET_ROOT; fallback to repo-level ../dataset
    dataset_root: Path = Path(os.getenv("DATASET_ROOT") or _default_dataset_root())

    # Video processing knobs
    # Threshold for window completeness when processing videos (0..1)
    video_completeness_threshold: float = float(os.getenv("VIDEO_COMPLETENESS", 0.8))
    # Override augmentation count for video windows (<=0 means use AUG_PER_SEQ)
    video_augment_per_seq: int = int(os.getenv("VIDEO_AUG_PER_SEQ", 0))
    # Skip intro frames until a hand is first detected
    video_skip_leading_no_hand: bool = bool(int(os.getenv("VIDEO_SKIP_LEADING_NO_HAND", 1)))
    # Stop processing after N consecutive frames with no detected hands (0 disables)
    video_stop_after_no_hand_frames: int = int(os.getenv("VIDEO_STOP_AFTER_NO_HAND_FRAMES", 0))
    # Comma-separated list of speed variants (e.g., "1.0,1.2,0.8")
    speed_variants_raw: str = os.getenv("SPEED_VARIANTS", "1.0,1.2,0.8")
    # Maximum number of saved samples per class per job
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

        # speed_variants is computed by validator; no assignment needed here

settings = Settings()
