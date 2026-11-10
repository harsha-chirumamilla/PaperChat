"""Grounded retrieval, reranking, and answer generation."""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from src.common.config import Config
from src.rag.query.llm_client import LLMClient
from src.rag.query.prompter import PromptFormatter
from src.rag.query.reranker import DocumentReranker
from src.rag.query.retriever import HybridRetriever
from src.rag.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class RAGSystem:
    """Standard RAG flow: hybrid retrieval, rank fusion, reranking, generation."""

    def __init__(self):
        logger.info("Initializing RAG system")
        self.llm_client = LLMClient(Config.MODEL_PROVIDER)
        self.vector_db = VectorDatabase()
        self.retriever = HybridRetriever(self.vector_db)
        self.reranker = DocumentReranker()
        self.prompter = PromptFormatter()

    def query(
        self,
        user_query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Answer a query and return citations from the same retrieval pass."""
        top_k = top_k or Config.RERANK_TOP_K
        retrieved = self.retriever.retrieve(
            user_query,
            filter_metadata=filter_metadata,
            k=max(Config.RETRIEVAL_K, top_k),
        )
        if not retrieved:
            return {
                "answer": "No relevant documents were found. Ingest papers before querying.",
                "citations": [],
            }

        reranked = self.reranker.rerank(user_query, retrieved, top_k=top_k)
        prompt = self.prompter.format_rag_prompt(user_query, reranked)
        answer = self.llm_client.generate(prompt)
        return {"answer": answer, "citations": self._citations(reranked)}

    @staticmethod
    def _citations(documents: List[Document]) -> List[Dict[str, Any]]:
        citations = []
        seen = set()
        for doc in documents:
            source = str(doc.metadata.get("source") or doc.metadata.get("arxiv_id") or "unknown")
            if source in seen:
                continue
            seen.add(source)
            snippet = doc.page_content[:200]
            citations.append({
                "number": len(citations) + 1,
                "title": doc.metadata.get("title", "Unknown"),
                "arxiv_id": source,
                "url": doc.metadata.get("url", ""),
                "snippet": snippet + ("..." if len(doc.page_content) > len(snippet) else ""),
            })
        return citations
