from celery import Celery
from app.config import settings

# dùng Redis làm broker & backend từ environment variables
celery_app = Celery(
    "sign_dataset",
    broker=settings.broker_url or "redis://redis:6379/0",
    backend=settings.result_backend or "redis://redis:6379/0",
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
)

# Import tasks to register them with Celery
from app import tasks
