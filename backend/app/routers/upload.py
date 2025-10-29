from fastapi import APIRouter, UploadFile, File, Form
import shutil
import os
import uuid

from app.processing import storage_utils as su
from app.tasks import enqueue_process_video
from fastapi import Body
import numpy as np

router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = "dataset/raw_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/video")
async def upload_video(
    file: UploadFile = File(...),
    user: str = Form(""),
    label: str = Form(...),
    dialect: str = Form(""),
    session_id: str = Form(None),
):
    if not session_id:
        session_id = uuid.uuid4().hex

    class_idx, folder = su.register_label(label)

    save_name = f"{user}_{label}_{uuid.uuid4().hex[:8]}_{file.filename}"
    file_path = os.path.join(UPLOAD_DIR, save_name)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Gửi task tới Celery
    job = enqueue_process_video.delay(video_path=file_path, user=user, label=label, session_id=session_id, dialect=dialect)

    # Normalize response to frontend UploadResult shape
    return {"success": True, "id": job.id, "session_id": session_id, "message": "queued"}


@router.post("/camera")
async def upload_camera(payload: dict = Body(...)):
    """
    Accept frames (array of arrays) and metadata, save as npz via storage_utils.save_sample
    Payload example: { user: str, label: str, session_id: str, dialect: str, frames: [{timestamp, landmarks}, ...] }
    """
    user = payload.get("user", "")
    label = payload.get("label")
    dialect = payload.get("dialect", "")
    session_id = payload.get("session_id", None) or uuid.uuid4().hex
    frames = payload.get("frames")

    if not label or not frames:
        return {"success": False, "message": "Missing label or frames"}

    # Ensure label exists
    class_idx, folder = su.register_label(label)

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
        pad = np.zeros((target_T - T, D))
        seq_padded = np.vstack([seq, pad])
    else:
        seq_padded = seq[:target_T]
    
    # Generate augmented sequences
    augmented_seq_list = generate_augmented_sequences(seq_padded)
    
    saved_paths = []
    for i, aseq in enumerate(augmented_seq_list):
        # Safety checks before saving
        if not isinstance(aseq, np.ndarray) or aseq.dtype.kind not in ("f", "i") or aseq.ndim != 2:
            print(f"[ERROR] Augmented sequence {i} not numeric 2D array: type={type(aseq)}, dtype={getattr(aseq, 'dtype', None)}, ndim={getattr(aseq, 'ndim', None)}")
            continue
            
        metadata = {
            "user": user, 
            "session_id": session_id, 
            "frames": target_T, 
            "source": "camera", 
            "dialect": dialect, 
            "created_at": su.now_str(),
            "augmented": True,
            "aug_index": i,
            "total_augs": len(augmented_seq_list)
        }
        
        path = su.save_sample(aseq, class_idx, folder, metadata=metadata)
        saved_paths.append(path)
    
    # Return multiple saved paths
    return {"success": True, "id": session_id, "paths": saved_paths, "total_samples": len(saved_paths), "message": f"saved {len(saved_paths)} augmented samples"}
