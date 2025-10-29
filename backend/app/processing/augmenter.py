"""
Refactored augmentation functions
- Stage A: Frame-level augmentation (flip, brightness, noise)
- Stage B: Keypoint-level augmentation (scaling, jitter, time-warp)
"""

import numpy as np
import cv2
import random

# -------- Stage A: Frame-level augment --------
def flip_frames(frames):
    return [cv2.flip(f, 1) for f in frames]

def add_gaussian_noise(frames, mean=0, sigma=10):
    noisy_frames = []
    for f in frames:
        noise = np.random.normal(mean, sigma, f.shape).astype(np.float32)
        noisy = np.clip(f.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        noisy_frames.append(noisy)
    return noisy_frames

def adjust_brightness(frames, factor=1.2):
    return [cv2.convertScaleAbs(f, alpha=factor, beta=0) for f in frames]

def stage_a_frame_level(frames):
    # apply a set of augmentations
    return {
        "original": frames,
        "flipped": flip_frames(frames),
        "bright": adjust_brightness(frames, 1.3),
        "noisy": add_gaussian_noise(frames, sigma=15)
    }

# -------- Stage B: Keypoint-level augment --------
def scale_sequence(seq: np.ndarray, scale_factor=1.1):
    return seq * scale_factor

def jitter_sequence(seq: np.ndarray, sigma=0.01):
    noise = np.random.normal(0, sigma, seq.shape)
    return seq + noise

def time_warp(seq: np.ndarray, factor=1.2):
    """simple temporal stretch/compress"""
    T, D = seq.shape
    new_T = int(T * factor)
    idx = np.linspace(0, T - 1, new_T).astype(np.int32)
    return seq[idx]

def stage_b_keypoint_level(seq: np.ndarray):
    return {
        "original": seq,
        "scaled_up": scale_sequence(seq, 1.1),
        "scaled_down": scale_sequence(seq, 0.9),
        "jittered_light": jitter_sequence(seq, 0.01),
        "jittered_heavy": jitter_sequence(seq, 0.03),
        "timewarp_fast": time_warp(seq, 1.3),
        "timewarp_slow": time_warp(seq, 0.8),
        "combined": jitter_sequence(scale_sequence(seq, 1.05), 0.015)
    }

# -------- Wrapper --------
def generate_augmented_sequences(sequence_array, config=None):
    # combine Stage B augmentations
    return list(stage_b_keypoint_level(sequence_array).values())
