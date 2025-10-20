from app.worker import celery_app
from app.processing.pipeline import process_video_job

@celery_app.task(bind=True)
def enqueue_process_video(self, video_path: str, user: str, label: str, session_id: str, dialect: str = ""):
    # This wrapper calls processing.pipeline (synchronous heavy processing)
    # Use try/except to capture failure and push status
    try:
        result = process_video_job(video_path, user, label, session_id, dialect)
        return {"status": "done", "result": result}
    except Exception as e:
        # you can log here and rethrow or return failure
        return {"status": "error", "error": str(e)}
