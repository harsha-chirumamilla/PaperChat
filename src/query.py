#!/usr/bin/env python3
"""
Query entry point for the GPU-Accelerated RAG Pipeline.
Queries the RAG system with a user question.
"""

import argparse
import sys
from src.rag.query.rag_pipeline import RAGSystem

def run_query(question: str):
    """
    Queries the RAG system with a user question.

    Args:
        question: User's question string
    """
    print(f"\n{'='*80}")
    print("QUERY MODE")
    print(f"{'='*80}\n")

    # Initialize RAG system
    rag = RAGSystem()

    # Get answer
    answer = rag.query(question)

    # Display results
    print(f"\n{'='*80}")
    print(f"QUESTION: {question}")
    print(f"{'='*80}\n")
    print(f"ANSWER:\n{answer}\n")
    print(f"{'='*80}\n")

    # Show citations
    citations = rag.get_citations(question)
    if citations:
        print("\nSOURCES:")
        for i, cite in enumerate(citations, 1):
            print(f"[{i}] {cite['title']}")
            print(f"arXiv: {cite['arxiv_id']} - {cite['url']}")
            print()

def main():
    parser = argparse.ArgumentParser(
        description="Query the GPU-Accelerated RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Query the system
        python query.py --question "What are attention mechanisms in transformers?"
                """
    )

    parser.add_argument('--question', type=str, required=True,
                        help='Question to ask the RAG system')

    args = parser.parse_args()

    run_query(args.question)

if __name__ == "__main__":
    main()