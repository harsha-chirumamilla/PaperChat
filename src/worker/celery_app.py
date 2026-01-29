import multiprocessing
import os
import logging
from celery import Celery
from celery.signals import worker_ready

logger = logging.getLogger(__name__)

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

@worker_ready.connect
def preload_models(sender=None, **kwargs):
    """
    Pre-load GPU models on worker startup to reduce first-query latency.
    This triggers lazy initialization of the RAG system before any tasks arrive.
    Saves 1-2 seconds on the first query by loading models during startup.
    """
    logger.info("Worker ready signal received - pre-warming GPU models...")
    try:
        from src.worker.tasks import get_rag_system
        
        # Trigger initialization (loads embedding model, reranker, BM25 index)
        rag_system = get_rag_system()
        logger.info("Worker pre-warmed successfully - models loaded")
    except Exception as e:
        logger.error("Failed to pre-warm worker: %s", e)
        # Don't crash worker - it can still lazy-load on first query
