import cv2, os
from app.processing.utils import ensure_dir

def sample_frames_from_video(video_path: str, target_fps: float = 5.0):
    """
    Decode video and sample frames roughly at target_fps.
    Returns list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    sample_rate = max(1, int(video_fps / target_fps))
    frames = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_rate == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames
