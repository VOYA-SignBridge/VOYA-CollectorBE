import logging
import os
from collections import deque
import numpy as np
from app.processing.ingest import frame_generator, temporal_speed_variants
from app.processing.keypoints_adapter import extract_sequence_stream
from app.processing.augmenter import generate_augmented_sequences
from app.config import settings
from app.dataset_manager import get_or_register_class
from app.dataset_samples import save_sequence_npz, count_samples_for_class

logger = logging.getLogger(__name__)


def _window_activity_mean_abs_diff(seq_arr: np.ndarray) -> float:
    """Compute a simple motion/activity score for a window.

    Returns mean absolute difference between consecutive frames across all features.
    Scale depends on feature representation (optionally normalized in keypoints_adapter).
    """
    if not isinstance(seq_arr, np.ndarray) or seq_arr.ndim != 2 or seq_arr.shape[0] < 2:
        return 0.0
    diffs = np.diff(seq_arr.astype(np.float32, copy=False), axis=0)
    return float(np.mean(np.abs(diffs)))

def process_video_job(video_path: str, user: str, label: str, session_id: str, dialect: str = "common", language: str = "vn"):
    """
    Synchronous function to process video without Celery decorator.
    This is called by the Celery task in tasks.py
    """
    try:
        fps_target = int(settings.fps_target)
        seq_len = int(settings.seq_len)
        stride = int(settings.stride)
        # Video can use a different augmentation multiplier than live capture
        video_aug = int(getattr(settings, "video_augment_per_seq", 0) or 0)
        aug_n = video_aug if video_aug > 0 else int(settings.augment_per_seq)
        resize = (int(settings.resize_width), int(settings.resize_height))

        # Create temporal speed variants from settings
        speed_variants = temporal_speed_variants(video_path, speeds=settings.speed_variants)
        
        # Resolve / register class metadata in new hierarchy
        class_meta = get_or_register_class(label_original=label, language=language, dialect=dialect or "")

        # Enforce a global per-class cap across the whole dataset (not just per upload job).
        # Set MAX_SAMPLES_PER_CLASS <= 0 to disable the cap.
        max_per_class = int(getattr(settings, "max_samples_per_class", 80))
        if max_per_class <= 0:
            existing_total = None
            remaining_quota = None
        else:
            existing_total = count_samples_for_class(class_meta.class_uid)
            remaining_quota = max(0, max_per_class - existing_total)
            if remaining_quota <= 0:
                logger.info(
                    "[CAP] class=%s already has %s samples (max=%s); skipping sample generation.",
                    class_meta.class_uid,
                    existing_total,
                    max_per_class,
                )
                return {
                    "status": "skipped_cap",
                    "reason": "class_cap_reached",
                    "saved": [],
                    "windows": 0,
                    "kept": 0,
                    "rejected": 0,
                    "variants": 0,
                    "cap": max_per_class,
                    "existing": existing_total,
                }
        
        all_saved_paths = []
        total_variants = len(speed_variants)
        global_kept = 0
        global_rejected = 0
        global_windows = 0
        saved_total = 0
        # Track best candidate windows so we can optionally top-up to a minimum sample count.
        # Each entry: (completeness, activity, seq_arr, window_meta)
        best_windows = []
        
        logger.info("[VIDEO] path=%s fps_target=%s resize=%sx%s variants=%s", 
                    video_path, fps_target, resize[0], resize[1], total_variants)

        last_valid_vec = np.zeros((settings.feature_dim,), dtype=np.float32)
        for speed_factor, variant_path in speed_variants:
            # Stream frames with optional resampling
            gen = frame_generator(variant_path, fps_target=fps_target, resize=resize)
            stream = extract_sequence_stream(gen)

            buffer = deque(maxlen=seq_len)
            hand_flags = deque(maxlen=seq_len)
            start_idx = None
            total_frames = 0
            total_windows = 0
            kept = 0
            rejected = 0
            saved_paths = []
            seen_hand = False
            trailing_no_hand = 0
            stop_after_no_hand = int(getattr(settings, "video_stop_after_no_hand_frames", 0) or 0)

            for idx, ts_idx, vec, has_hand in stream:
                if settings.debug_logging:
                    logger.debug("[FRAME] idx=%s has_hand=%s", idx, has_hand)
                total_frames += 1

                # Skip leading frames until the first detected hand appears.
                # This avoids labeling pre-action "idle" prefix as part of the gesture.
                if getattr(settings, "video_skip_leading_no_hand", True) and (not seen_hand) and (not has_hand):
                    continue
                if has_hand:
                    seen_hand = True
                    trailing_no_hand = 0
                elif seen_hand:
                    trailing_no_hand += 1
                    if stop_after_no_hand > 0 and trailing_no_hand >= stop_after_no_hand:
                        # Stop at the start of outro (no hands) to avoid generating windows
                        # dominated by trailing idle frames.
                        break

                if start_idx is None:
                    start_idx = idx
                # Carry-forward missing hand frames if enabled
                if not has_hand and settings.carry_forward_missing:
                    vec_cf = last_valid_vec.copy()
                    buffer.append(vec_cf)
                else:
                    buffer.append(vec)
                    if has_hand:
                        last_valid_vec = vec.astype(np.float32)
                hand_flags.append(bool(has_hand))

                if len(buffer) == seq_len:
                    total_windows += 1
                    completeness = float(sum(1 for f in hand_flags if f)) / float(seq_len)
                    # Build window array once (even if rejected) so we can optionally rescue later.
                    seq_arr = np.vstack(list(buffer)).astype(np.float32)
                    # ensure shape is (seq_len, feature_dim)
                    if seq_arr.shape != (seq_len, settings.feature_dim):
                        T, D = seq_arr.shape
                        if D < settings.feature_dim:
                            pad = np.zeros((T, settings.feature_dim - D), dtype=np.float32)
                            seq_arr = np.hstack([seq_arr, pad])
                        else:
                            seq_arr = seq_arr[:, :settings.feature_dim]

                    activity = _window_activity_mean_abs_diff(seq_arr)
                    activity_thr = float(getattr(settings, "video_activity_threshold", 0.0) or 0.0)
                    is_complete = completeness >= float(settings.video_completeness_threshold)
                    is_active = (activity_thr <= 0.0) or (activity >= activity_thr)

                    window_meta = {
                        "user": user,
                        "session_id": session_id,
                        "fps_original": fps_target,
                        "fps_processed": fps_target,
                        "speed_factor": float(speed_factor),
                        "start_frame": int(idx - seq_len + 1),
                        "end_frame": int(idx),
                        "completeness": float(round(completeness, 4)),
                        "activity": float(round(activity, 6)),
                        "created_at": None,
                    }

                    # Keep a small pool of best windows for best-effort minimum sample fill.
                    best_windows.append((float(completeness), float(activity), seq_arr, window_meta))
                    if len(best_windows) > 20:
                        best_windows.sort(key=lambda x: (x[0], x[1]), reverse=True)
                        best_windows = best_windows[:20]

                    if not (is_complete and is_active):
                        rejected += 1
                    else:
                        kept += 1
                        augmented_seq_list = generate_augmented_sequences(seq_arr, config={"n": aug_n})
                        for aug_id, aseq in enumerate(augmented_seq_list):
                            # Re-check global cap (best-effort) before each save.
                            if max_per_class > 0 and count_samples_for_class(class_meta.class_uid) >= max_per_class:
                                break
                            path = save_sequence_npz(class_meta, aseq, meta=window_meta, augment_id=aug_id, source_type="video")
                            saved_paths.append(path)
                            saved_total += 1
                            if remaining_quota is not None and saved_total >= remaining_quota:
                                break

                    # slide by stride
                    for _ in range(min(stride, len(buffer))):
                        if buffer:
                            buffer.popleft()
                        if hand_flags:
                            hand_flags.popleft()

                # Stop early if we've reached the remaining quota for this class
                if remaining_quota is not None and saved_total >= remaining_quota:
                    break

            # Cleanup temp speed variant file
            if variant_path != video_path and os.path.exists(variant_path):
                try:
                    os.remove(variant_path)
                except:
                    pass

            logger.info("[VARIANT] speed=%.1fx frames=%s windows=%s kept=%s rejected=%s saved=%s",
                        speed_factor, total_frames, total_windows, kept, rejected, len(saved_paths))
            
            all_saved_paths.extend(saved_paths)
            global_kept += kept
            global_rejected += rejected
            global_windows += total_windows

            # Stop processing further variants if cap reached
            if remaining_quota is not None and saved_total >= remaining_quota:
                logger.info(
                    "[CAP] Reached global max_samples_per_class=%s (existing=%s + saved=%s); stopping further variants.",
                    max_per_class,
                    existing_total,
                    saved_total,
                )
                break

        # Best-effort: ensure each video upload yields at least N samples (subject to remaining quota).
        min_target = int(getattr(settings, "video_min_samples_per_video", 0) or 0)
        if remaining_quota is not None:
            min_target = max(0, min(min_target, remaining_quota))
        else:
            min_target = max(0, min_target)
        if min_target > 0 and saved_total < min_target and best_windows:
            best_windows.sort(key=lambda x: (x[0], x[1]), reverse=True)
            need = min_target - saved_total
            logger.info(
                "[MIN] saved=%s < min_target=%s; best-effort fill need=%s.",
                saved_total,
                min_target,
                need,
            )

            best_comp, best_act, best_seq, best_meta = best_windows[0]
            fill_n = need if remaining_quota is None else min(need, remaining_quota - saved_total)
            if fill_n > 0:
                fill_variants = generate_augmented_sequences(best_seq, config={"n": fill_n})
                base_aug_id = 10_000
                for i, aseq in enumerate(fill_variants):
                    if saved_total >= min_target:
                        break
                    if remaining_quota is not None and saved_total >= remaining_quota:
                        break
                    if max_per_class > 0 and count_samples_for_class(class_meta.class_uid) >= max_per_class:
                        break
                    meta = dict(best_meta)
                    meta["rescue_fill"] = True
                    meta["rescue_from_completeness"] = float(round(best_comp, 4))
                    meta["rescue_from_activity"] = float(round(best_act, 6))
                    path = save_sequence_npz(class_meta, aseq, meta=meta, augment_id=base_aug_id + i, source_type="video")
                    all_saved_paths.append(path)
                    saved_total += 1

        logger.info("[SEQ] total_windows=%s kept=%s rejected=%s", global_windows, global_kept, global_rejected)
        logger.info("[SAVE] class=%s saved=%s augmented=%s per_seq=%s output=%s",
                label, len(all_saved_paths), global_kept * aug_n, aug_n, class_meta.hierarchy_path())

        return {"status": "success", "saved": all_saved_paths, "windows": global_windows, 
                "kept": global_kept, "rejected": global_rejected, "variants": total_variants}

    except Exception as e:
        raise Exception(f"Pipeline processing failed: {str(e)}")
