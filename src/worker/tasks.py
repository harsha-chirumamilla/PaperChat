from src.worker.celery_app import celery_app
from src.rag.query.rag_pipeline import RAGSystem
from src.rag.ingestion.data_loaders import ArxivLoader
from src.rag.ingestion.pdf_processor import PDFProcessor
from src.rag.vector_db import VectorDatabase
import os
import threading
import logging
from pathlib import Path

import redis
import json
import hashlib

logger = logging.getLogger(__name__)

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
                logger.info("Initializing RAG System in worker")
                try:
                    _rag_system = RAGSystem()
                except Exception as e:
                    logger.error("Failed to initialize RAG System: %s", e)
                    raise
    
    return _rag_system

@celery_app.task(bind=True)
def process_query(self, query: str):
    rag_system = get_rag_system()
    
    # Check cache
    cache_key = f"query:{hashlib.md5(query.encode()).hexdigest()}"
    cached_result = redis_client.get(cache_key)
    if cached_result:
        logger.info("Cache hit for query: %s", query)
        return json.loads(cached_result)

    logger.info("Processing query: %s", query)
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
def ingest_documents(self, query: str = "cat:cs.AI", max_results: int = 5, skip_papers: int = 0):
    logger.info("Ingesting documents for query: %s (max: %d, skip: %d)", query, max_results, skip_papers)
    try:
        loader = ArxivLoader()
        papers = loader.fetch_papers(query=query, max_results=max_results, skip_papers=skip_papers)
        
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
        logger.error("Ingestion failed: %s", e)
        raise e

@celery_app.task(bind=True)
def process_uploaded_pdf(self, file_path: str, filename: str):
    """
    Process an uploaded PDF file and add it to the vector database.
    
    Args:
        file_path: Path to the uploaded PDF file
        filename: Original filename
        
    Returns:
        Processing result with status and statistics
    """
    logger.info("Processing uploaded PDF: %s", filename)
    try:
        # Extract text from PDF
        text = PDFProcessor.extract_text(file_path)
        
        # Create metadata for uploaded paper
        paper_id = f"upload:{Path(filename).stem}"
        paper_metadata = {
            "id": paper_id,
            "title": Path(filename).stem,
            "source": "uploaded",
            "arxiv_id": "N/A",
            "url": "N/A",
            "filepath": file_path
        }
        
        # Create chunks
        chunks = PDFProcessor.create_chunks(text, paper_metadata)
        
        if chunks:
            # Add to vector database
            db = VectorDatabase()
            db.add_documents(chunks)
            
            logger.info("Successfully processed uploaded PDF: %s (%d chunks)", filename, len(chunks))
            return {
                "status": "success",
                "filename": filename,
                "chunks_added": len(chunks),
                "message": "Paper successfully processed and added to database"
            }
        else:
            logger.warning("No content extracted from uploaded PDF: %s", filename)
            return {
                "status": "no_content",
                "filename": filename,
                "chunks_added": 0,
                "message": "No text content could be extracted from the PDF"
            }
            
    except Exception as e:
        logger.error("Failed to process uploaded PDF %s: %s", filename, e)
        raise e

@celery_app.task(bind=True)
def generate_suggested_questions(self, current_query: str, answer: str, chat_history: list):
    """
    Generate suggested follow-up questions based on the current conversation.
    
    Args:
        current_query: The current user query
        answer: The answer that was provided
        chat_history: List of previous Q&A pairs
        
    Returns:
        List of suggested follow-up questions
    """
    logger.info("Generating suggested questions for query: %s", current_query)
    try:
        rag_system = get_rag_system()
        
        # Build context from chat history
        history_context = ""
        if chat_history:
            history_context = "Previous conversation:\n"
            for item in chat_history[-3:]:  # Use last 3 exchanges
                history_context += f"Q: {item.get('query', '')}\nA: {item.get('answer', '')[:200]}...\n\n"
        
        # Create prompt for generating suggestions
        prompt = f"""{history_context}Current Question: {current_query}

Current Answer: {answer[:500]}...

Based on the conversation above, suggest 3 insightful follow-up questions that would help the user explore this topic deeper. The questions should:
1. Build on the current context
2. Be specific and answerable from research papers
3. Explore different angles (clarification, deeper dive, related topics)

Format: Return ONLY the questions, one per line, without numbering or bullets."""
        
        # Generate suggestions using LLM
        suggestions_text = rag_system.llm_client.generate(prompt)
        
        # Parse suggestions into list
        suggestions = [
            line.strip().lstrip('123456789.- ')
            for line in suggestions_text.split('\n')
            if line.strip() and len(line.strip()) > 10
        ]
        
        # Limit to 3 suggestions
        suggestions = suggestions[:3]
        
        logger.info("Generated %d suggested questions", len(suggestions))
        return {
            "status": "success",
            "suggestions": suggestions
        }
        
    except Exception as e:
        logger.error("Failed to generate suggested questions: %s", e)
        raise e
