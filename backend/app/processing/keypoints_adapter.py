"""
Refactored keypoints extraction from collect_dataset.py
- Extract Mediapipe Hands landmarks only
- Flatten into fixed-length vector
"""

from typing import List, Tuple, Generator
import numpy as np
import mediapipe as mp
import cv2
from app.config import settings
from app.processing.utils import canonicalize_hands_126


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    """
    Normalize a flattened 126-dim keypoint vector:
    - Reshape to (2, 21, 3)
    - Determine reference point (wrist of first detected hand or mean of non-zero points)
    - Translate so reference is at origin, then scale by max span (x/y) to normalize size
    Returns flattened vector of same shape.
    """
    if vec is None:
        return vec
    try:
        arr = vec.reshape(2, 21, 3).astype(np.float32)
    except Exception:
        return vec

    coords = arr.reshape(-1, 3)
    # mask non-zero landmarks
    mask = (coords.sum(axis=1) != 0)
    if not mask.any():
        return vec

    coords_nonzero = coords[mask][:, :2]

    # prefer wrist (landmark 0) of first detected hand
    wrist = None
    for h in range(2):
        w = arr[h, 0, :2]
        if not np.allclose(w, 0.0):
            wrist = w
            break
    if wrist is None:
        wrist = coords_nonzero.mean(axis=0)

    # translate
    coords[:, :2] = coords[:, :2] - wrist

    # compute scale (max span across non-zero points)
    xs = coords_nonzero[:, 0] - wrist[0]
    ys = coords_nonzero[:, 1] - wrist[1]
    span_x = xs.max() - xs.min() if xs.size else 0.0
    span_y = ys.max() - ys.min() if ys.size else 0.0
    scale = max(span_x, span_y)
    if scale <= 1e-6:
        scale = 1.0

    coords[:, :2] = coords[:, :2] / float(scale)

    return coords.reshape(-1)

# constants for hands only
N_HAND = 21

def extract_sequence_from_frames(frames: List[np.ndarray], config: dict = None):
    """
    frames: list of BGR images
    return: np.ndarray shape (T, D) where D = 2 hands * 21 landmarks * 3 coords = 126
    """
    mp_hands = mp.solutions.hands
    seq = []
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        for frame in frames:
            # Optionally mirror input frames (some frontends/cameras mirror preview)
            if getattr(settings, "mirror_input", False):
                frame = cv2.flip(frame, 1)
            img_rgb = frame[:, :, ::-1]
            results = hands.process(img_rgb)
            kp_dict = extract_keypoints_from_results(results)
            vec = flatten_keypoints(kp_dict)
            # Optional normalization of keypoints to center+scale invariance
            if getattr(settings, "normalize_keypoints", False):
                vec = _normalize_vector(vec)
            # Canonicalize for mirror/hand invariance
            if getattr(settings, "canonicalize_hands", True):
                vec = canonicalize_hands_126(
                    vec,
                    normalized=bool(getattr(settings, "normalize_keypoints", False)),
                    mirror_invariant=bool(getattr(settings, "canonicalize_mirror", True)),
                )
            seq.append(vec)
    if len(seq) == 0:
        return np.zeros((0, 126), dtype=np.float32)  # 2 hands * 21 * 3 = 126
    return np.stack(seq, axis=0)


def detect_and_vectorize(frame: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Run MediaPipe Hands on a single frame and return (flattened 126-dim vector, has_hand flag).
    """
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        if getattr(settings, "mirror_input", False):
            frame = cv2.flip(frame, 1)
        img_rgb = frame[:, :, ::-1]
        results = hands.process(img_rgb)
        kp_dict = extract_keypoints_from_results(results)
        vec = flatten_keypoints(kp_dict)
        if getattr(settings, "normalize_keypoints", False):
            vec = _normalize_vector(vec)
        if getattr(settings, "canonicalize_hands", True):
            vec = canonicalize_hands_126(
                vec,
                normalized=bool(getattr(settings, "normalize_keypoints", False)),
                mirror_invariant=bool(getattr(settings, "canonicalize_mirror", True)),
            )
        has_hand = bool(np.any(vec != 0.0))
        return vec.astype(np.float32), has_hand


def extract_sequence_stream(frame_iter: Generator[Tuple[int, float, np.ndarray], None, None]) -> Generator[Tuple[int, int, np.ndarray, bool], None, None]:
    """
    Given a frame generator (idx, ts, frame), yield (idx, ts_idx, vector, has_hand) per frame.
    ts_idx is idx for now; caller can compute timestamps if needed.
    """
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        for idx, ts, frame in frame_iter:
            if getattr(settings, "mirror_input", False):
                frame = cv2.flip(frame, 1)
            img_rgb = frame[:, :, ::-1]
            results = hands.process(img_rgb)
            kp_dict = extract_keypoints_from_results(results)
            vec = flatten_keypoints(kp_dict)
            if getattr(settings, "normalize_keypoints", False):
                vec = _normalize_vector(vec)
            if getattr(settings, "canonicalize_hands", True):
                vec = canonicalize_hands_126(
                    vec,
                    normalized=bool(getattr(settings, "normalize_keypoints", False)),
                    mirror_invariant=bool(getattr(settings, "canonicalize_mirror", True)),
                )
            has_hand = bool(np.any(vec != 0.0))
            yield idx, int(idx), vec.astype(np.float32), has_hand

def extract_keypoints_from_results(results):
    """
    Extract hand landmarks from MediaPipe Hands results
    Returns left and right hand keypoints (or zeros if not detected)
    """
    def lm_to_list(landmarks, expected_n):
        if not landmarks:
            return np.zeros((expected_n, 3), dtype=np.float32)
        coords = []
        for i in range(expected_n):
            if i < len(landmarks.landmark):
                lm = landmarks.landmark[i]
                coords.append([lm.x, lm.y, getattr(lm, "z", 0.0)])
            else:
                coords.append([0.0, 0.0, 0.0])
        return np.array(coords, dtype=np.float32)

    # Initialize hands as empty
    left_hand = np.zeros((N_HAND, 3), dtype=np.float32)
    right_hand = np.zeros((N_HAND, 3), dtype=np.float32)
    
    # Extract hand landmarks if detected
    if results.multi_hand_landmarks and results.multi_handedness:
        for i, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            # Determine if it's left or right hand
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            hand_keypoints = lm_to_list(hand_landmarks, N_HAND)
            
            if hand_label == "Left":
                left_hand = hand_keypoints
            elif hand_label == "Right":
                right_hand = hand_keypoints

    return {
        "left_hand": left_hand,
        "right_hand": right_hand
    }

def flatten_keypoints(kp_dict):
    """
    Flatten hand keypoints only
    Returns: vector of size 126 (2 hands * 21 landmarks * 3 coords)
    """
    left = kp_dict["left_hand"].flatten()   # 21 * 3 = 63
    right = kp_dict["right_hand"].flatten() # 21 * 3 = 63
    return np.concatenate([left, right], axis=0)  # Total: 126
