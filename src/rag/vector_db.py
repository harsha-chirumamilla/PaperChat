from typing import List
import logging
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument
from src.common.config import Config
from src.rag.embeddings import get_embedding_function

logger = logging.getLogger(__name__)


class VectorDatabase:
    def __init__(self, persistence_directory: str = Config.DB_DIR):
        self.embedding_fn = get_embedding_function()
        self.persist_dir = persistence_directory
        self._db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embedding_fn,
            collection_name="arxiv_papers"
        )
        
    def add_documents(self, documents: List[LangChainDocument]):
        """Adds documents to the vector store."""
        if not documents:
            return
        logger.info("Adding %d chunks to vector database", len(documents))
        self._db.add_documents(documents)
        logger.info("Successfully added documents to database")
        
    def as_retriever(self, k: int = Config.RETRIEVAL_K):
        return self._db.as_retriever(search_kwargs={"k": k})
    
    def similarity_search(self, query: str, k: int = Config.RETRIEVAL_K, filter: dict = None) -> List[LangChainDocument]:
        """
        Perform similarity search with optional metadata filtering.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            filter: Optional metadata filter (e.g., {'section': 'methods'})
            
        Returns:
            List of similar documents
        """
        return self._db.similarity_search(query, k=k, filter=filter)
    
    def get_all_documents(self) -> List[LangChainDocument]:
        """Retrieves all documents from the vector store (for BM25 init)."""
        data = self._db.get()
        docs = []
        if data and data['documents']:
            for i, text in enumerate(data['documents']):
                meta = data['metadatas'][i] if data['metadatas'] else {}
                docs.append(LangChainDocument(page_content=text, metadata=meta))
        return docs

    def collection_version(self) -> int:
        """Return the current chunk count so BM25 can refresh after ingestion."""
        return int(self._db._collection.count())
