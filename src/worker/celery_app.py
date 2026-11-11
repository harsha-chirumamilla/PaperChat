import os
from celery import Celery

# Get Redis URL from env or default
BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
RESULT_BACKEND_URL = os.getenv("CELERY_RESULT_BACKEND", BROKER_URL)

celery_app = Celery(
    "paperchat",
    broker=BROKER_URL,
    backend=RESULT_BACKEND_URL,
    include=["src.worker.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    task_track_started=True,
    result_expires=86400,
)
