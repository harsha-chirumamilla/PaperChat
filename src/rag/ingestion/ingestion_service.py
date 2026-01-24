"""
Ingestion service for the GPU-Accelerated RAG Pipeline.
Contains the main ingestion workflow as a reusable function.
"""

from src.rag.ingestion.data_loaders import ArxivLoader
from src.rag.ingestion.pdf_processor import PDFProcessor
from src.rag.vector_db import VectorDatabase


def run_ingestion(category: str, max_papers: int):
    """
    Downloads papers from arXiv, processes them, and stores in vector DB.

    Args:
        category: arXiv category query (e.g., "cat:cs.AI")
        max_papers: Maximum number of papers to fetch
    """
    print(f"\n{'='*80}")
    print(f"INGESTION MODE: Fetching {max_papers} papers from '{category}'")
    print(f"{'='*80}\n")

    # Step 1: Download papers
    loader = ArxivLoader()
    papers = loader.fetch_papers(query=category, max_results=max_papers)

    if not papers:
        print("No papers fetched. Exiting.")
        return

    print(f"\nFetched {len(papers)} papers")

    # Step 2: Process PDFs and create chunks
    all_chunks = []
    for paper in papers:
        print(f"\nProcessing: {paper['title'][:60]}...")

        text = PDFProcessor.extract_text(paper['filepath'])

        if not text.strip():
            print(f"No text extracted from {paper['id']}. Skipping.")
            continue

        chunks = PDFProcessor.create_chunks(text, paper)
        all_chunks.extend(chunks)
        print(f"Generated {len(chunks)} chunks")

    print(f"\nTotal chunks created: {len(all_chunks)}")

    # Step 3: Add to vector database
    db = VectorDatabase()
    db.add_documents(all_chunks)

    print(f"\n{'='*80}")
    print("INGESTION COMPLETE")
    print(f"{'='*80}\n")
