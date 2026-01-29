from typing import List, Dict, Optional
import logging
import concurrent.futures
import pickle
import os
import hashlib
import json
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from src.common.config import Config
from src.rag.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retrieval combining BM25 and vector search with Redis caching."""
    
    def __init__(self, vector_db: VectorDatabase, redis_client=None):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_db: Vector database instance
            redis_client: Optional Redis client for caching retrieval results
        """
        self.vector_db = vector_db
        self.redis = redis_client
        self.chroma_retriever = vector_db.as_retriever(k=Config.RETRIEVAL_K)
        self._init_bm25_retriever()
    
    def _init_bm25_retriever(self):
        """Initializes BM25 retriever from cached index or builds new one."""
        index_path = os.path.join(Config.DB_DIR, "bm25_index.pkl")
        
        # Try loading from disk cache
        if os.path.exists(index_path):
            try:
                logger.info("Loading BM25 index from disk cache")
                with open(index_path, 'rb') as f:
                    self.bm25_retriever = pickle.load(f)
                self.bm25_retriever.k = Config.RETRIEVAL_K
                logger.info("BM25 index loaded successfully")
                return
            except Exception as e:
                logger.warning("Failed to load BM25 cache: %s. Rebuilding...", e)
        
        # Build from scratch
        logger.info("Building BM25 index from documents")
        docs = self.vector_db.get_all_documents()

        if docs:
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = Config.RETRIEVAL_K
            
            # Persist to disk
            try:
                with open(index_path, 'wb') as f:
                    pickle.dump(self.bm25_retriever, f)
                logger.info("BM25 index cached to disk")
            except Exception as e:
                logger.warning("Failed to cache BM25 index: %s", e)
        else:
            logger.warning("No documents found in database. BM25 retriever disabled")
            self.bm25_retriever = None
    
    def invalidate_bm25_cache(self):
        """Invalidate BM25 cache. Call after adding new documents."""
        index_path = os.path.join(Config.DB_DIR, "bm25_index.pkl")
        if os.path.exists(index_path):
            try:
                os.remove(index_path)
                logger.info("BM25 cache invalidated")
            except Exception as e:
                logger.warning("Failed to invalidate BM25 cache: %s", e)
    
    def _vector_retrieve(self, query: str, filter_metadata: Dict = None) -> List[Document]:
        """Vector retrieval helper (for parallel execution)."""
        if filter_metadata:
            return self.vector_db.similarity_search(
                query,
                k=Config.RETRIEVAL_K,
                filter=filter_metadata
            )
        else:
            return self.chroma_retriever.invoke(query)
    
    def _bm25_retrieve(self, query: str, filter_metadata: Dict = None) -> List[Document]:
        """BM25 retrieval helper (for parallel execution)."""
        if not self.bm25_retriever:
            return []
        
        bm25_docs = self.bm25_retriever.invoke(query)
        
        # Apply metadata filter if specified
        if filter_metadata:
            bm25_docs = [
                doc for doc in bm25_docs
                if all(doc.metadata.get(k) == v for k, v in filter_metadata.items())
            ]
        
        return bm25_docs
    
    def retrieve(
        self,
        query: str,
        filter_metadata: Dict = None
    ) -> List[Document]:
        """
        Hybrid retrieval: BM25 + Vector search with optional metadata filtering and Redis caching.
        Runs BM25 and vector search in parallel for better performance.
        
        Args:
            query: Search query
            filter_metadata: Optional metadata filters (e.g., {'section': 'methods'})
            
        Returns:
            List of retrieved documents
        """
        # Check Redis cache (only for non-filtered queries)
        if self.redis and not filter_metadata:
            cache_key = f"retrieval:{hashlib.md5(query.encode()).hexdigest()}"
            try:
                cached = self.redis.get(cache_key)
                if cached:
                    logger.info("Retrieval cache hit for query: %s", query[:50])
                    data = json.loads(cached)
                    return [
                        Document(page_content=d['content'], metadata=d['metadata'])
                        for d in data
                    ]
            except Exception as e:
                logger.warning("Cache read error: %s", e)
        
        scored_docs: Dict[str, Dict] = {}

        # Parallel retrieval for 30-40% speed improvement
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both retrieval tasks
            vector_future = executor.submit(self._vector_retrieve, query, filter_metadata)
            bm25_future = executor.submit(self._bm25_retrieve, query, filter_metadata)
            
            # Gather results
            vector_docs = vector_future.result()
            bm25_docs = bm25_future.result()
        
        # Combine scores from vector search
        for doc in vector_docs:
            key = doc.page_content
            if key not in scored_docs:
                scored_docs[key] = {"doc": doc, "score": 0.0}
            scored_docs[key]["score"] += 0.5

        # Combine scores from BM25
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

        final_docs = [item["doc"] for item in ranked[:Config.RETRIEVAL_K]]
        
        # Cache results (5 minute TTL for non-filtered queries)
        if self.redis and not filter_metadata:
            try:
                cache_data = [
                    {'content': doc.page_content, 'metadata': doc.metadata}
                    for doc in final_docs
                ]
                self.redis.setex(cache_key, 300, json.dumps(cache_data))
            except Exception as e:
                logger.warning("Cache write error: %s", e)
        
        return final_docs
