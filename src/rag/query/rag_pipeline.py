from typing import List, Dict

from src.common.config import Config
from src.rag.vector_db import VectorDatabase
from src.rag.query.retriever import HybridRetriever
from src.rag.query.reranker import DocumentReranker
from src.rag.query.prompter import PromptFormatter
from src.rag.query.llm_client import LLMClient


class RAGSystem:
    def __init__(self):
        """
        Initializes the RAG system with configurable API provider (OpenAI, Anthropic, or Gemini),
        Hybrid Retrieval (BM25 + Vector), and CrossEncoder reranking.
        """
        print("Initializing RAG System...")
        
        # Initialize LLM client
        self.llm_client = LLMClient(Config.MODEL_PROVIDER)

        # Initialize vector database
        self.vector_db = VectorDatabase()

        # Initialize retriever
        self.retriever = HybridRetriever(self.vector_db)

        # Initialize reranker
        self.reranker = DocumentReranker()
        
        # Initialize prompt formatter
        self.prompter = PromptFormatter()

        print("RAG System initialized.")

    def query(self, user_query: str) -> str:
        """
        Full RAG pipeline: Hybrid Retrieve → Rerank → Generate
        
        Args:
            user_query: User's question
            
        Returns:
            Generated answer
        """
        print("Retrieving documents...")
        retrieved_docs = self.retriever.retrieve(user_query)

        if not retrieved_docs:
            return (
                "No relevant documents found in the database. "
                "Please ingest papers first."
            )

        reranked_docs = self.reranker.rerank(user_query, retrieved_docs)

        prompt = self.prompter.format_rag_prompt(user_query, reranked_docs)

        print(f"Generating answer with {self.llm_client.provider} ({self.llm_client.model_name})...")
        return self.llm_client.generate(prompt)

    def get_citations(self, user_query: str) -> List[Dict]:
        """
        Returns citations for a given query.
        
        Args:
            user_query: User's question
            
        Returns:
            List of citation dictionaries
        """
        retrieved_docs = self.retriever.retrieve(user_query)
        reranked_docs = self.reranker.rerank(user_query, retrieved_docs)

        citations = []
        for doc in reranked_docs:
            citations.append(
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "arxiv_id": doc.metadata.get("source", "N/A"),
                    "url": doc.metadata.get("url", "N/A"),
                    "snippet": doc.page_content[:200] + "...",
                }
            )

        return citations
