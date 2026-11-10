"""Hybrid dense and BM25 retrieval with reciprocal-rank fusion."""

import hashlib
import logging
import threading
from typing import Any, Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.common.config import Config
from src.rag.vector_db import VectorDatabase

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Retrieve with dense and sparse search, then combine rankings with RRF."""

    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
        self._bm25_retriever = None
        self._bm25_collection_version = None
        self._bm25_lock = threading.Lock()
        self._bm25_docs: List[Document] = []

    @staticmethod
    def _document_key(doc: Document) -> str:
        source = doc.metadata.get("source") or doc.metadata.get("arxiv_id")
        chunk_id = doc.metadata.get("chunk_id")
        if source is not None and chunk_id is not None:
            return f"{source}:{chunk_id}"
        return hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()

    def _ensure_bm25(self, collection_version: int) -> None:
        if self._bm25_collection_version == collection_version:
            return
        with self._bm25_lock:
            if self._bm25_collection_version == collection_version:
                return
            self._bm25_docs = self.vector_db.get_all_documents()
            self._bm25_retriever = (
                BM25Retriever.from_documents(self._bm25_docs) if self._bm25_docs else None
            )
            self._bm25_collection_version = collection_version
            logger.info("Built BM25 index for %d chunks", len(self._bm25_docs))

    def _bm25_search(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]],
    ) -> List[Document]:
        docs = self._bm25_docs
        if filter_metadata:
            docs = [
                doc for doc in docs
                if all(doc.metadata.get(key) == value for key, value in filter_metadata.items())
            ]
            retriever = BM25Retriever.from_documents(docs) if docs else None
        else:
            retriever = self._bm25_retriever
        if retriever is None:
            return []
        retriever.k = k
        return retriever.invoke(query)

    def retrieve(
        self,
        query: str,
        filter_metadata: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None,
    ) -> List[Document]:
        k = k or Config.RETRIEVAL_K
        collection_version = self.vector_db.collection_version()
        self._ensure_bm25(collection_version)

        dense_docs = self.vector_db.similarity_search(query, k=k, filter=filter_metadata)
        sparse_docs = self._bm25_search(query, k, filter_metadata)

        # Reciprocal Rank Fusion is robust when dense and sparse scores use different scales.
        fused: Dict[str, Dict[str, Any]] = {}
        for ranking in (dense_docs, sparse_docs):
            for rank, doc in enumerate(ranking, start=1):
                key = self._document_key(doc)
                if key not in fused:
                    fused[key] = {"doc": doc, "score": 0.0}
                fused[key]["score"] += 1.0 / (60 + rank)

        return [entry["doc"] for entry in sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:k]]
