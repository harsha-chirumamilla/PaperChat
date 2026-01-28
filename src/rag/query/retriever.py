from typing import List, Dict
import logging
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.common.config import Config
from src.rag.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining BM25 and vector search."""
    
    def __init__(self, vector_db: VectorDatabase):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_db: Vector database instance
        """
        self.vector_db = vector_db
        self.chroma_retriever = vector_db.as_retriever(k=Config.RETRIEVAL_K)
        self._init_bm25_retriever()
    
    def _init_bm25_retriever(self):
        """Initializes BM25 retriever from all stored documents."""
        logger.info("Initializing BM25 retriever")
        docs = self.vector_db.get_all_documents()

        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = Config.RETRIEVAL_K
        else:
            logger.warning("No documents found in database. BM25 retriever disabled")
            self.bm25_retriever = None
    
    def retrieve(
        self,
        query: str,
        filter_metadata: Dict = None
    ) -> List[Document]:
        """
        Hybrid retrieval: BM25 + Vector search with optional metadata filtering.
        
        Args:
            query: Search query
            filter_metadata: Optional metadata filters (e.g., {'section': 'methods'})
            
        Returns:
            List of retrieved documents
        """
        scored_docs: Dict[str, Dict] = {}

        # Vector retrieval with filters
        if filter_metadata:
            # Use vector DB similarity search with filter
            vector_docs = self.vector_db.similarity_search(
                query,
                k=Config.RETRIEVAL_K,
                filter=filter_metadata
            )
        else:
            vector_docs = self.chroma_retriever.invoke(query)
        
        for doc in vector_docs:
            key = doc.page_content
            if key not in scored_docs:
                scored_docs[key] = {"doc": doc, "score": 0.0}
            scored_docs[key]["score"] += 0.5

        # BM25 retrieval (filter after retrieval since BM25 doesn't support metadata filters easily)
        if self.bm25_retriever:
            bm25_docs = self.bm25_retriever.invoke(query)
            
            # Apply metadata filter if specified
            if filter_metadata:
                bm25_docs = [
                    doc for doc in bm25_docs
                    if all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
                ]
            
            for doc in bm25_docs:
                key = doc.page_content
                if key not in scored_docs:
                    scored_docs[key] = {"doc": doc, "score": 0.0}
                scored_docs[key]["score"] += 0.5

        # Sort by combined score
        ranked = sorted(
            scored_docs.values(),
            key=lambda x: x["score"],
            reverse=True,
        )

        return [item["doc"] for item in ranked[:Config.RETRIEVAL_K]]
