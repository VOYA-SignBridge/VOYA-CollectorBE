from app.processing.ingest import sample_frames_from_video
from app.processing.keypoints_adapter import extract_sequence_from_frames
from app.processing.augmenter import generate_augmented_sequences
from app.processing import storage_utils as su
import numpy as np
import os

def process_video_job(video_path: str, user: str, label: str, session_id: str, dialect: str = ""):
    """
    Synchronous function to process video without Celery decorator.
    This is called by the Celery task in tasks.py
    """
    try:
        frames = sample_frames_from_video(video_path, target_fps=6.0)
        if not frames:
            raise RuntimeError("No frames extracted")

        seq = extract_sequence_from_frames(frames)
        if seq.size == 0:
            raise RuntimeError("No keypoints extracted")

        T, D = seq.shape
        target_T = 60
        if T < target_T:
            pad = np.zeros((target_T - T, D))
            seq_padded = np.vstack([seq, pad])
        else:
            seq_padded = seq[:target_T]

        augmented_seq_list = generate_augmented_sequences(seq_padded)

        class_idx, folder = su.register_label(label)
        saved_paths = []
        for aseq in augmented_seq_list:
            meta = {"user": user, "session_id": session_id, "frames": target_T, "source": "video", "dialect": dialect}
            path = su.save_sample(aseq, class_idx, folder, metadata=meta)
            saved_paths.append(path)

        return {"status": "success", "saved": saved_paths}

    except Exception as e:
        raise Exception(f"Pipeline processing failed: {str(e)}")
