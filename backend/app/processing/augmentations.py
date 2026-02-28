import numpy as np

from app.config import settings


def _scale(seq: np.ndarray, factor: float) -> np.ndarray:
    return (seq * factor).astype(np.float32)


def _jitter(seq: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, seq.shape).astype(np.float32)
    return (seq + noise).astype(np.float32)


def _time_warp(seq: np.ndarray, factor: float) -> np.ndarray:
    T, D = seq.shape
    new_T = max(1, int(round(T * factor)))
    idx = np.linspace(0, T - 1, new_T).astype(np.int32)
    warped = seq[idx]
    # ensure same length by pad/truncate back to T
    if warped.shape[0] < T:
        pad = np.zeros((T - warped.shape[0], D), dtype=np.float32)
        warped = np.vstack([warped, pad])
    else:
        warped = warped[:T]
    return warped.astype(np.float32)


def _hand_dropout(seq: np.ndarray, drop_left: bool = True) -> np.ndarray:
    # left hand occupies first 63 dims, right hand last 63 dims
    out = seq.copy().astype(np.float32)
    if drop_left:
        out[:, :63] = 0.0
    else:
        out[:, 63:] = 0.0
    return out


def _mirror_hands(seq: np.ndarray) -> np.ndarray:
    """
    Proper horizontal mirror for keypoint sequences.
    This does two things:
    - Flip the x coordinate (normalized) for all landmarks: x -> 1 - x
    - Swap left/right blocks so the resulting order matches flipped input
    """
    out = seq.copy().astype(np.float32)
    if out.size == 0:
        return out
    # left and right blocks (each 21 landmarks * 3 coords = 63)
    T = out.shape[0]
    left = out[:, :63].reshape(T, 21, 3).copy()
    right = out[:, 63:].reshape(T, 21, 3).copy()

    # flip x coordinate (first coord in each landmark triple)
    # - raw MediaPipe coords are in [0,1] -> mirror via x := 1-x
    # - normalized coords (centered) -> mirror via x := -x
    if bool(getattr(settings, "normalize_keypoints", False)):
        left[..., 0] = -left[..., 0]
        right[..., 0] = -right[..., 0]
    else:
        left[..., 0] = 1.0 - left[..., 0]
        right[..., 0] = 1.0 - right[..., 0]

    # swap hands (left <- right, right <- left) after flipping x
    out[:, :63] = right.reshape(T, 63)
    out[:, 63:] = left.reshape(T, 63)
    return out


def augment_n(seq: np.ndarray, n: int = 8) -> list:
    """
    Produce up to n augmentations deterministically composed from common ops
    to match live-capture distribution.
    Always returns a list of length n (duplicates original if needed).
    """
    variants = []
    variants.append(seq)  # original
    variants.append(_jitter(seq, 0.01))
    variants.append(_jitter(seq, 0.02))
    variants.append(_scale(seq, 1.05))
    variants.append(_scale(seq, 0.95))
    variants.append(_time_warp(seq, 1.15))
    variants.append(_time_warp(seq, 0.85))
    variants.append(_hand_dropout(seq, drop_left=True))
    variants.append(_hand_dropout(seq, drop_left=False))
    # If we canonicalize mirror orientation at ingestion/inference time, adding a mirrored
    # variant often becomes a near-duplicate. Keep it only when mirror canonicalization is off.
    if not bool(getattr(settings, "canonicalize_mirror", True)):
        variants.append(_mirror_hands(seq))

    # ensure uniform length n; keep first n
    if len(variants) >= n:
        return variants[:n]
    else:
        # pad with copies of seq
        while len(variants) < n:
            variants.append(seq)
        return variants
