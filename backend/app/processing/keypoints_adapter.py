"""
Refactored keypoints extraction from collect_dataset.py
- Extract Mediapipe Holistic landmarks
- Flatten into fixed-length vector
"""

from typing import List
import numpy as np
import mediapipe as mp

# constants (giống file collect_dataset.py bạn gửi)
N_POSE = 25   # upper body
N_HAND = 21
N_FACE = 468

def extract_sequence_from_frames(frames: List[np.ndarray], config: dict = None):
    """
    frames: list of BGR images
    return: np.ndarray shape (T, D)
    """
    mp_holistic = mp.solutions.holistic
    seq = []
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:
        for frame in frames:
            img_rgb = frame[:, :, ::-1]
            results = holistic.process(img_rgb)
            kp_dict = extract_keypoints_from_results(results)
            vec = flatten_keypoints(kp_dict)
            seq.append(vec)
    if len(seq) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.stack(seq, axis=0)

def extract_keypoints_from_results(results):
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

    return {
        "pose": lm_to_list(results.pose_landmarks, N_POSE),
        "left_hand": lm_to_list(results.left_hand_landmarks, N_HAND),
        "right_hand": lm_to_list(results.right_hand_landmarks, N_HAND),
        "face": lm_to_list(results.face_landmarks, N_FACE)
    }

def flatten_keypoints(kp_dict):
    # flatten order: pose, left, right, face
    pose = kp_dict["pose"].flatten()
    left = kp_dict["left_hand"].flatten()
    right = kp_dict["right_hand"].flatten()
    face = kp_dict["face"].flatten()
    return np.concatenate([pose, left, right, face], axis=0)
