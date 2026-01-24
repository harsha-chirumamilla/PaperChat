"""
RAG (Retrieval-Augmented Generation) package.

This package is organized into:
- embeddings: Shared embedding utilities  
- vector_db: ChromaDB vector database operations
- ingestion: Data loading, PDF processing, and document ingestion
- query: LLM client, retrieval, reranking, and RAG pipeline
"""

# Re-export commonly used classes for convenience
from src.rag.vector_db import VectorDatabase
from src.rag.embeddings import get_device, get_embedding_function

__all__ = [
    "VectorDatabase",
    "get_device", 
    "get_embedding_function",
]
