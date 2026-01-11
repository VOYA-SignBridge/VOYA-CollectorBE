import cv2, os, subprocess, shutil, tempfile, logging
from typing import Generator, Tuple
from app.processing.utils import ensure_dir
from app.config import settings

logger = logging.getLogger(__name__)

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


def ffmpeg_resample(input_path: str, fps_target: int) -> Tuple[str, float, float]:
    """
    If ffmpeg is available, resample video to fixed fps before processing.
    Returns (output_path, fps_in, fps_out). On failure or missing ffmpeg, returns (input_path, fps_in, fps_in).
    """
    # probe fps_in via OpenCV
    cap = cv2.VideoCapture(input_path)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
    cap.release()

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path or fps_in == float(fps_target):
        return input_path, float(fps_in or 0.0), float(fps_in or 0.0)

    # Create temp output in same directory for atomicity/cleanup
    dir_name = os.path.dirname(input_path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix="resampled_", suffix=".mp4", dir=dir_name)
    os.close(fd)
    cmd = [
        ffmpeg_path,
        "-y",
        "-i", input_path,
        "-filter:v", f"fps={fps_target}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-an",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if settings.debug_logging:
            logger.debug("[RESAMPLE] fps_in=%.2f fps_out=%d path=%s", fps_in, fps_target, tmp_path)
        return tmp_path, float(fps_in or 0.0), float(fps_target)
    except Exception as e:
        if settings.debug_logging:
            logger.debug("[RESAMPLE][FAIL] using original file. error=%s", e)
        # fallback on failure
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return input_path, float(fps_in or 0.0), float(fps_in or 0.0)


def frame_generator(video_path: str,
                    fps_target: int = None,
                    resize: Tuple[int, int] = None) -> Generator[Tuple[int, float, 'cv2.Mat'], None, None]:
    """
    Stream frames from a (possibly resampled) video, yielding (frame_idx, timestamp_sec, frame_bgr).
    Performs optional resize for speed. Uses target FPS to compute timestamps.
    """
    fps_target = fps_target or settings.fps_target
    # Resample if needed using ffmpeg
    resampled_path, fps_in, fps_out = ffmpeg_resample(video_path, fps_target)

    cap = cv2.VideoCapture(resampled_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    # Use fps_out when available; fallback to cap's reported fps
    fps = fps_out or cap.get(cv2.CAP_PROP_FPS) or float(fps_target)
    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize and resize[0] > 0 and resize[1] > 0:
                frame = cv2.resize(frame, resize)
            ts = idx / float(fps)
            if settings.debug_logging and idx % 50 == 0:
                logger.debug("[FRAME_GEN] idx=%d ts=%.3f", idx, ts)
            yield idx, ts, frame
            idx += 1
    finally:
        cap.release()
        # cleanup temp resampled file if created
        if resampled_path != video_path and os.path.exists(resampled_path):
            try:
                os.remove(resampled_path)
            except Exception:
                pass


def temporal_speed_variants(video_path: str, speeds=[1.0, 1.2, 0.8]) -> list:
    """
    Create temporal speed variants of video using ffmpeg setpts filter.
    Returns list of (speed_factor, temp_video_path).
    """
    variants = []
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        # Fallback: only original speed
        return [(1.0, video_path)]
    
    dir_name = os.path.dirname(video_path) or "."
    
    for speed in speeds:
        if speed == 1.0:
            variants.append((speed, video_path))
            continue
        
        # Create temp variant
        fd, tmp_path = tempfile.mkstemp(prefix=f"speed{speed}_", suffix=".mp4", dir=dir_name)
        os.close(fd)
        
        # setpts formula: slower = larger multiplier, faster = smaller
        pts_multiplier = 1.0 / speed
        cmd = [
            ffmpeg_path, "-y", "-i", video_path,
            "-filter:v", f"setpts={pts_multiplier}*PTS",
            "-an", tmp_path
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            variants.append((speed, tmp_path))
            if settings.debug_logging:
                logger.debug("[SPEED_VARIANT] speed=%.2f path=%s", speed, tmp_path)
        except Exception as e:
            if settings.debug_logging:
                logger.debug("[SPEED_VARIANT][FAIL] speed=%.2f error=%s", speed, e)
            try:
                os.remove(tmp_path)
            except:
                pass
    
    return variants
