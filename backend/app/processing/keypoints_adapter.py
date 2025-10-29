"""
Refactored keypoints extraction from collect_dataset.py
- Extract Mediapipe Hands landmarks only
- Flatten into fixed-length vector
"""

from typing import List
import numpy as np
import mediapipe as mp

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
            img_rgb = frame[:, :, ::-1]
            results = hands.process(img_rgb)
            kp_dict = extract_keypoints_from_results(results)
            vec = flatten_keypoints(kp_dict)
            seq.append(vec)
    if len(seq) == 0:
        return np.zeros((0, 126), dtype=np.float32)  # 2 hands * 21 * 3 = 126
    return np.stack(seq, axis=0)

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
