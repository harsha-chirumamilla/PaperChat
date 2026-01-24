import multiprocessing
import os
from celery import Celery

# CRITICAL: Set spawn method BEFORE any multiprocessing use
# This must be at module level, before Celery initialization
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, that's fine
    pass

# Get Redis URL from env or default
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "paperchat",
    broker=REDIS_URL,
    backend=REDIS_URL,
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
    # task_track_started=True,
)
