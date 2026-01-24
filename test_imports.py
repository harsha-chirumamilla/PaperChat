#!/usr/bin/env python3
"""Verify that all imports work correctly in the restructured codebase."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")
print("=" * 80)

try:
    print("\n1. Testing shared utilities...")
    from src.rag.embeddings import get_device, get_embedding_function
    from src.rag.vector_db import VectorDatabase
    print("   ✓ Shared utilities imported successfully")
    
    print("\n2. Testing ingestion module...")
    from src.rag.ingestion.data_loaders import ArxivLoader
    from src.rag.ingestion.pdf_processor import PDFProcessor
    from src.rag.ingestion.ingestion_service import run_ingestion
    print("   ✓ Ingestion module imported successfully")
    
    print("\n3. Testing query module...")
    from src.rag.query.llm_client import LLMClient
    from src.rag.query.retriever import HybridRetriever
    from src.rag.query.reranker import DocumentReranker
    from src.rag.query.prompter import PromptFormatter
    from src.rag.query.rag_pipeline import RAGSystem
    print("   ✓ Query module imported successfully")
    
    print("\n4. Testing external entry points...")
    from src.worker.tasks import process_query, ingest_documents
    print("   ✓ Worker tasks imported successfully")
    
    print("\n" + "=" * 80)
    print("✓ All imports verified successfully!")
    print("=" * 80)
    
except ImportError as e:
    print(f"\n✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
