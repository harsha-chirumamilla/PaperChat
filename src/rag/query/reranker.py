"""Cross-encoder reranking for hybrid retrieval results."""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.common.config import Config
from src.rag.embeddings import get_device

logger = logging.getLogger(__name__)


class DocumentReranker:
    def __init__(self):
        device = get_device()
        logger.info("Loading reranker %s on %s", Config.RERANKER_MODEL_ID, device)
        self.reranker = CrossEncoder(Config.RERANKER_MODEL_ID, device=device)
        if device == "cuda":
            try:
                self.reranker.model = self.reranker.model.half()
            except Exception:
                logger.exception("Could not enable half precision for reranker")

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        if not documents:
            return []
        scores = self.reranker.predict([[query, doc.page_content] for doc in documents])
        ranked = sorted(zip(documents, scores), key=lambda item: item[1], reverse=True)
        return [doc for doc, _ in ranked[: (top_k or Config.RERANK_TOP_K)]]
