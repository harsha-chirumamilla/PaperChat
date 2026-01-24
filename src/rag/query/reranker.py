from typing import List
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from src.common.config import Config
from src.rag.embeddings import get_device


class DocumentReranker:
    """Document reranking using CrossEncoder."""
    
    def __init__(self):
        """Initialize reranker with CrossEncoder model."""
        print(f"Loading Reranker: {Config.RERANKER_MODEL_ID}")
        device = get_device()
        self.reranker = CrossEncoder(
            Config.RERANKER_MODEL_ID,
            device=device
        )
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Reranks documents using a CrossEncoder.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []

        print(f"Reranking {len(documents)} documents...")
        pairs = [[query, doc.page_content] for doc in documents]
        scores = self.reranker.predict(pairs)

        ranked = sorted(
            zip(documents, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [doc for doc, _ in ranked[:Config.RERANK_TOP_K]]
