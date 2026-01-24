from worker.celery_app import celery_app
from src.rag.query import RAGSystem
from src.rag.ingestion import ArxivLoader, PDFProcessor
from src.rag import VectorDatabase
import os
import threading

import redis
import json
import hashlib

# Connect to Redis (same as Celery broker)
REDIS_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# Global RAG system with lazy initialization
_rag_system = None
_rag_lock = threading.Lock()

def get_rag_system():
    """Lazy-load RAGSystem on first use to avoid CUDA fork issues."""
    global _rag_system
    
    if _rag_system is None:
        with _rag_lock:
            # Double-check locking pattern
            if _rag_system is None:
                print("Initializing RAG System in worker...")
                try:
                    _rag_system = RAGSystem()
                except Exception as e:
                    print(f"Failed to initialize RAG System: {e}")
                    raise
    
    return _rag_system

@celery_app.task(bind=True)
def process_query(self, query: str):
    rag_system = get_rag_system()
    
    # Check cache
    cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        print(f"Cache hit for query: {query}")
        return json.loads(cached_result)

    print(f"Processing query: {query}")
    answer = rag_system.query(query)
    citations = rag_system.get_citations(query)
    
    result = {
        "answer": answer,
        "citations": citations
    }
    
    # Cache result (expire in 1 hour)
    redis_client.setex(cache_key, 3600, json.dumps(result))
    
    return result

@celery_app.task(bind=True)
def ingest_documents(self, query: str = "cat:cs.AI", max_results: int = 5):
    print(f"Ingesting documents for query: {query}")
    try:
        loader = ArxivLoader()
        papers = loader.fetch_papers(query=query, max_results=max_results)
        
        all_chunks = []
        for paper in papers:
            text = PDFProcessor.extract_text(paper["filepath"])
            chunks = PDFProcessor.create_chunks(text, paper)
            all_chunks.extend(chunks)
            
        if all_chunks:
            db = VectorDatabase()
            db.add_documents(all_chunks)
            return {"status": "success", "papers_ingested": len(papers), "chunks_added": len(all_chunks)}
        else:
            return {"status": "no_content", "papers_ingested": len(papers), "chunks_added": 0}
            
    except Exception as e:
        print(f"Ingestion failed: {e}")
        raise e
