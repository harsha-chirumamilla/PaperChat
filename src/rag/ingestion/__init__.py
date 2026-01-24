"""
Ingestion package for RAG pipeline.

Handles downloading arXiv papers, processing PDFs, and ingesting documents into the vector database.
"""

from src.rag.ingestion.data_loaders import ArxivLoader
from src.rag.ingestion.pdf_processor import PDFProcessor
from src.rag.ingestion.ingestion_service import run_ingestion

__all__ = [
    "ArxivLoader",
    "PDFProcessor", 
    "run_ingestion",
]

