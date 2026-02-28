from fastapi import APIRouter, UploadFile, File, Form
import shutil
import os
import uuid
import time
import logging

from app.processing import storage_utils as su  # legacy for timestamp helper
from app.dataset_manager import get_or_register_class, normalize_dialect
from app.dataset_samples import save_sequence_npz
from app.tasks import enqueue_process_video
from fastapi import Body
import numpy as np
from app.config import settings
from app.processing.utils import canonicalize_hands_126

router = APIRouter(prefix="/upload", tags=["upload"])

# Align raw video storage with DATASET_ROOT so it matches features path
UPLOAD_DIR = str(settings.dataset_root / "raw_videos")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/video")
async def upload_video(
    file: UploadFile = File(...),
    user: str = Form(""),
    label: str = Form(...),
    language: str = Form("vn"),
    dialect: str = Form("common"),
    session_id: str = Form(None),
):
    start = time.time()
    log = logging.getLogger("upload.video")
    if not session_id:
        session_id = uuid.uuid4().hex

    # Normalize dialect input from FE
    dialect = normalize_dialect(dialect)
    language = (language or "vn").lower().strip()

    log.info("[UPLOAD][video] user=%s label=%s lang=%s dialect=%s filename=%s session=%s", user, label, language, dialect, getattr(file, 'filename', ''), session_id)

    # Register / fetch class in new hierarchy
    class_meta = get_or_register_class(label_original=label, language=language, dialect=dialect or "")

    save_name = f"{user}_{label}_{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, save_name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # Log the resolved save path and dataset root for debugging
    log.info("[UPLOAD][video] saved path=%s dataset_root=%s", file_path, settings.dataset_root)

    # Gửi task tới Celery
    try:
        job = enqueue_process_video.delay(video_path=file_path, user=user, label=label, session_id=session_id, dialect=dialect, language=language)
        log.info("[UPLOAD][video] queued job=%s elapsed=%.3fs", getattr(job, 'id', 'unknown'), time.time() - start)
        return {"success": True, "id": job.id, "session_id": session_id, "message": "queued"}
    except Exception as e:
        log.error("[UPLOAD][video][ERROR] queue failed: %s", e)
        return {"success": False, "message": f"queue failed: {e}"}
    


@router.post("/camera")
async def upload_camera(payload: dict = Body(...)):
    """
    Accept frames (array of arrays) and metadata, save as npz via storage_utils.save_sample
    Payload example: { user: str, label: str, session_id: str, dialect: str, frames: [{timestamp, landmarks}, ...] }
    """
    user = payload.get("user", "")
    label = payload.get("label")
    dialect = normalize_dialect(payload.get("dialect", "common"))
    language = payload.get("language", "vn")
    session_id = payload.get("session_id", None) or uuid.uuid4().hex
    frames = payload.get("frames")

    if not label or not frames:
        return {"success": False, "message": "Missing label or frames"}

    # Ensure label exists
    class_meta = get_or_register_class(label_original=label, language=language, dialect=dialect or "")

    # Convert frames (list of {timestamp, landmarks}) into numpy array
    # We expect landmarks arrays per frame; stack into (T, N) array
    try:
        # helper: convert a MediaPipe-like dict into a flat numeric vector (hands only)
        def flatten_landmarks(ld):
            # If already a list/array of numbers, return as-is
            if ld is None:
                return None
            if isinstance(ld, (list, tuple, np.ndarray)):
                return np.asarray(ld)

            # If dict (MediaPipe style) with keys for hands only
            if isinstance(ld, dict):
                parts = []
                # Only process hands (left_hand, right_hand) - no pose, no face
                for key in ("left_hand", "right_hand"):
                    elems = ld.get(key, [])
                    # each elem is expected to be dict with x,y,z (no visibility for hands)
                    for p in elems:
                        if p is None:
                            # missing point -> pad zeros (only x,y,z for hands)
                            parts.extend([0.0, 0.0, 0.0])
                            continue
                        x = p.get("x") if isinstance(p, dict) else None
                        y = p.get("y") if isinstance(p, dict) else None
                        z = p.get("z") if isinstance(p, dict) else None
                        # Only x,y,z for hands (no visibility)
                        parts.extend([
                            float(x) if x is not None else 0.0,
                            float(y) if y is not None else 0.0,
                            float(z) if z is not None else 0.0,
                        ])
                return np.array(parts, dtype="float32")

            # Unknown format -> attempt to coerce
            return np.asarray(ld)

        landmarks_seq = []
        for f in frames:
            raw = f.get("landmarks")
            flat = flatten_landmarks(raw)
            if flat is None:
                raise ValueError("frame missing landmarks")
            landmarks_seq.append(flat)

        # Ensure all frames have same vector length by padding shorter ones
        maxlen = max([a.size for a in landmarks_seq])
        # Build a numeric 2D array explicitly to avoid object-dtype pitfalls
        T = len(landmarks_seq)
        seq = np.zeros((T, maxlen), dtype="float32")
        for i, a in enumerate(landmarks_seq):
            if a.size > maxlen:
                # truncate if unexpectedly longer
                seq[i, :] = a[:maxlen].astype("float32")
            else:
                seq[i, : a.size] = a.astype("float32")

        # Debug output
        print(f"[DEBUG] First frame landmarks type: {type(frames[0].get('landmarks'))}")
        print(f"[DEBUG] First frame landmarks shape/content: {frames[0].get('landmarks')}")
        print(f"[DEBUG] Built numeric sequence shape: {seq.shape}, dtype: {seq.dtype}")
        # Ensure sequence is numeric float32 (some inputs may produce object-dtype rows)
        try:
            seq = seq.astype("float32")
        except Exception as e:
            print(f"[WARN] seq.astype failed: {e}, attempting per-row conversion")
            new = np.zeros((T, maxlen), dtype="float32")
            for i in range(T):
                row = landmarks_seq[i]
                try:
                    arr = np.asarray(row, dtype=np.float32).flatten()
                except Exception:
                    # best-effort flatten for nested dict/list structures
                    vals = []
                    def collect(x):
                        if x is None:
                            return
                        if isinstance(x, (int, float)):
                            vals.append(float(x))
                        elif isinstance(x, dict):
                            # prefer x,y,z,visibility order if available
                            for k in ("x", "y", "z", "visibility"):
                                if k in x:
                                    try:
                                        vals.append(float(x.get(k) or 0.0))
                                    except Exception:
                                        vals.append(0.0)
                            # if dict has nested lists, collect them too
                            for v in x.values():
                                if isinstance(v, (list, tuple)):
                                    for it in v:
                                        collect(it)
                        elif isinstance(x, (list, tuple, np.ndarray)):
                            for it in x:
                                collect(it)
                        else:
                            # ignore unknown types
                            return
                    collect(row)
                    arr = np.asarray(vals, dtype=np.float32)

                if arr.size > maxlen:
                    new[i, :] = arr[:maxlen]
                else:
                    new[i, : arr.size] = arr
            seq = new
    except Exception as e:
        print(f"[ERROR] Error processing landmarks: {e}")
        return {"success": False, "message": f"Invalid frames payload: {e}"}

    # Apply augmentation to create multiple samples
    from app.processing.augmenter import generate_augmented_sequences
    
    # Ensure sequence has proper shape (pad to 60 frames)
    T, D = seq.shape
    target_T = 60
    if T < target_T:
        pad = np.zeros((target_T - T, D), dtype=np.float32)
        seq_padded = np.vstack([seq, pad])
    else:
        seq_padded = seq[:target_T]

    # Ensure feature dimension matches spec (hands-only = 126)
    from app.config import settings as _settings
    feat = int(getattr(_settings, "feature_dim", 126))
    tT, tD = seq_padded.shape
    if tD < feat:
        col_pad = np.zeros((tT, feat - tD), dtype=np.float32)
        seq_padded = np.hstack([seq_padded.astype(np.float32), col_pad])
    elif tD > feat:
        seq_padded = seq_padded[:, :feat].astype(np.float32)
    else:
        seq_padded = seq_padded.astype(np.float32)

    # Canonicalize per-frame for mirror/hand invariance (useful when FE is mirrored)
    if getattr(settings, "canonicalize_hands", True) and seq_padded.shape[1] == 126:
        mirror_invariant = bool(getattr(settings, "canonicalize_mirror", True))
        for t in range(seq_padded.shape[0]):
            seq_padded[t, :] = canonicalize_hands_126(
                seq_padded[t, :],
                normalized=False,
                mirror_invariant=mirror_invariant,
            )
    
    # Generate augmented sequences
    augmented_seq_list = generate_augmented_sequences(seq_padded)
    
    saved_paths = []
    for i, aseq in enumerate(augmented_seq_list):
        # Safety checks before saving
        if not isinstance(aseq, np.ndarray) or aseq.dtype.kind not in ("f", "i") or aseq.ndim != 2:
            print(f"[ERROR] Augmented sequence {i} not numeric 2D array: type={type(aseq)}, dtype={getattr(aseq, 'dtype', None)}, ndim={getattr(aseq, 'ndim', None)}")
            continue
            
        meta = {
            "user": user,
            "session_id": session_id,
            "fps_original": None,
            "fps_processed": None,
            "completeness": None,
            "created_at": su.now_str(),
        }
        path = save_sequence_npz(class_meta, aseq, meta=meta, augment_id=i, source_type="camera")
        saved_paths.append(path)
    
    # Return multiple saved paths
    return {"success": True, "id": session_id, "paths": saved_paths, "total_samples": len(saved_paths), "message": f"saved {len(saved_paths)} augmented samples", "language": language, "dialect": dialect}
