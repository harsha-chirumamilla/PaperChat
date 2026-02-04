"""
Ingestion service for the GPU-Accelerated RAG Pipeline.
Contains the main ingestion workflow as a reusable function.
"""

import logging
from src.rag.ingestion.data_loaders import ArxivLoader
from src.rag.ingestion.pdf_processor import PDFProcessor
from src.rag.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


def run_ingestion(category: str, max_papers: int, skip_papers: int = 0):
    """
    Downloads papers from arXiv, processes them, and stores in vector DB.

    Args:
        category: arXiv category query (e.g., "cat:cs.AI")
        max_papers: Maximum number of papers to fetch
        skip_papers: Number of papers to skip (for pagination)
    """
    logger.info("="*80)
    logger.info("INGESTION MODE: Fetching %d papers from '%s' (skip: %d)", max_papers, category, skip_papers)
    logger.info("="*80)

    # Step 1: Download papers
    loader = ArxivLoader()
    papers = loader.fetch_papers(query=category, max_results=max_papers, skip_papers=skip_papers)

    if not papers:
        logger.warning("No papers fetched. Exiting")
        return

    logger.info("Fetched %d papers", len(papers))

    # Step 2: Process PDFs and create chunks
    all_chunks = []
    for paper in papers:
        logger.info("Processing: %s", paper['title'][:60])

        text = PDFProcessor.extract_text(paper['filepath'])

        if not text.strip():
            logger.warning("No text extracted from %s. Skipping", paper['id'])
            continue

        chunks = PDFProcessor.create_chunks(text, paper)
        all_chunks.extend(chunks)
        logger.info("Generated %d chunks", len(chunks))

    logger.info("Total chunks created: %d", len(all_chunks))

    # Step 3: Add to vector database
    db = VectorDatabase()
    db.add_documents(all_chunks)

    logger.info("="*80)
    logger.info("INGESTION COMPLETE")
    logger.info("="*80)
