#!/usr/bin/env python3
"""
CLI entry point for the GPU-Accelerated RAG Pipeline ingestion.
"""

import argparse
from src.rag.ingestion import run_ingestion
from src.common.config import Config

def main():
    parser = argparse.ArgumentParser(
        description="Ingestion for GPU-Accelerated RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Ingest 10 papers from cs.AI
        python ingestion_entrypoint.py --category "cat:cs.AI" --max-papers 10
                """
    )

    parser.add_argument('--category', type=str, default='cat:cs.AI',
                        help='arXiv category query (default: cat:cs.AI)')
    parser.add_argument('--max-papers', type=int, default=Config.MAX_PAPERS_PER_BATCH,
                        help=f'Maximum papers to fetch (default: {Config.MAX_PAPERS_PER_BATCH})')

    args = parser.parse_args()
    run_ingestion(args.category, args.max_papers)

if __name__ == "__main__":
    main()