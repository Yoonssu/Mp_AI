from celery import Celery

from app.config import get_settings


settings = get_settings()

celery_app = Celery(
    "mp_ai",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/1",
)

celery_app.conf.update(
    task_track_started=True,
    timezone="Asia/Seoul",
)
