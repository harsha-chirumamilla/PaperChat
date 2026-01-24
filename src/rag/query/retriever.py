from typing import List, Dict
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.common.config import Config
from src.rag.vector_db import VectorDatabase


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
        print("Initializing BM25 Retriever...")
        docs = self.vector_db.get_all_documents()

        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = Config.RETRIEVAL_K
        else:
            print("Warning: No documents found. BM25 disabled.")
            self.bm25_retriever = None
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Hybrid retrieval: BM25 + Vector search.
        
        Args:
            query: Search query
            
        Returns:
            List of retrieved documents
        """
        scored_docs: Dict[str, Dict] = {}

        # BM25 retrieval
        if self.bm25_retriever:
            for doc in self.bm25_retriever.invoke(query):
                key = doc.page_content
                if key not in scored_docs:
                    scored_docs[key] = {"doc": doc, "score": 0.0}
                scored_docs[key]["score"] += 0.5

        # Vector retrieval
        for doc in self.chroma_retriever.invoke(query):
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
