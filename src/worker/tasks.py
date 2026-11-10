"""Background tasks for API queries and document ingestion."""

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import redis

from src.common.config import Config
from src.rag.ingestion.data_loaders import ArxivLoader
from src.rag.ingestion.pdf_processor import PDFProcessor
from src.rag.query.rag_pipeline import RAGSystem
from src.rag.vector_db import VectorDatabase
from src.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
_rag_system = None
_rag_lock = threading.Lock()


def get_rag_system() -> RAGSystem:
    """Lazy-load model instances inside each worker process."""
    global _rag_system
    if _rag_system is None:
        with _rag_lock:
            if _rag_system is None:
                _rag_system = RAGSystem()
    return _rag_system


def _query_cache_key(
    query: str,
    top_k: int,
    filter_metadata: Optional[Dict[str, Any]],
    corpus_version: str,
) -> str:
    payload = {
        "schema": 2,
        "query": " ".join(query.split()),
        "top_k": top_k,
        "filter": filter_metadata or {},
        "corpus_version": corpus_version,
        "provider": Config.MODEL_PROVIDER,
        "model": {
            "openai": Config.OPENAI_MODEL_ID,
            "anthropic": Config.ANTHROPIC_MODEL_ID,
            "gemini": Config.GEMINI_MODEL_ID,
        }.get(Config.MODEL_PROVIDER),
        "reranker": Config.RERANKER_MODEL_ID,
        "retrieval_k": Config.RETRIEVAL_K,
        "temperature": Config.TEMPERATURE,
        "max_new_tokens": Config.MAX_NEW_TOKENS,
        "embedding_model": Config.EMBEDDING_MODEL_ID,
        "prompt_version": "grounded-citations-v1",
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"answer:v2:{digest}"


def _corpus_version() -> str:
    try:
        return redis_client.get("paperchat:corpus_version") or "0"
    except Exception:
        logger.exception("Could not read corpus version; answer caching may be unavailable")
        return "redis-unavailable"


def _mark_corpus_updated() -> None:
    try:
        redis_client.incr("paperchat:corpus_version")
    except Exception:
        logger.exception("Could not advance corpus version after ingestion")


@celery_app.task(bind=True)
def process_query(
    self,
    query: str,
    top_k: Optional[int] = None,
    use_cache: bool = True,
    filter_metadata: Optional[Dict[str, Any]] = None,
):
    top_k = top_k or Config.RERANK_TOP_K
    corpus_version = _corpus_version()
    cache_key = _query_cache_key(query, top_k, filter_metadata, corpus_version)

    if use_cache:
        try:
            cached = redis_client.get(cache_key)
            if cached:
                logger.info("Exact answer cache hit")
                return json.loads(cached)
        except Exception:
            logger.exception("Answer cache read failed; generating a fresh answer")

    result = get_rag_system().query(query, top_k=top_k, filter_metadata=filter_metadata)
    if use_cache:
        try:
            redis_client.setex(cache_key, Config.CACHE_TTL_SECONDS, json.dumps(result))
        except Exception:
            logger.exception("Answer cache write failed")
    return result


def _index_papers(papers):
    chunks = []
    for paper in papers:
        text = PDFProcessor.extract_text(paper["filepath"])
        if text.strip():
            chunks.extend(PDFProcessor.create_chunks(text, paper))
    if not chunks:
        return {"status": "no_content", "papers_ingested": len(papers), "chunks_added": 0}
    VectorDatabase().add_documents(chunks)
    _mark_corpus_updated()
    return {"status": "success", "papers_ingested": len(papers), "chunks_added": len(chunks)}


@celery_app.task(bind=True)
def ingest_documents(self, query: str = "cat:cs.AI", max_results: int = 5, skip_papers: int = 0):
    logger.info("Ingesting arXiv papers: query=%s, max=%d, skip=%d", query, max_results, skip_papers)
    papers = ArxivLoader().fetch_papers(query=query, max_results=max_results, skip_papers=skip_papers)
    return _index_papers(papers)


@celery_app.task(bind=True)
def process_uploaded_pdf(self, file_path: str, filename: str):
    logger.info("Processing uploaded PDF: %s", filename)
    try:
        text = PDFProcessor.extract_text(file_path)
        metadata = {
            "id": f"upload:{Path(filename).stem}",
            "title": Path(filename).stem,
            "source": "uploaded",
            "arxiv_id": "N/A",
            "url": "",
            "filepath": file_path,
        }
        chunks = PDFProcessor.create_chunks(text, metadata) if text.strip() else []
        if not chunks:
            return {"status": "no_content", "filename": filename, "chunks_added": 0}
        VectorDatabase().add_documents(chunks)
        _mark_corpus_updated()
        return {"status": "success", "filename": filename, "chunks_added": len(chunks)}
    finally:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not remove uploaded file %s", file_path)
