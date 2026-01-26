from typing import List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document as LangChainDocument
from src.common.config import Config
from src.rag.embeddings import get_device


class VectorDatabase:
    def __init__(self, persistence_directory: str = Config.DB_DIR):
        device = get_device()
        self.embedding_fn = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_ID,
            model_kwargs={'device': device}, 
            encode_kwargs={'normalize_embeddings': True}
        )
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
        print(f"Adding {len(documents)} chunks to vector database...")
        self._db.add_documents(documents)
        print("Done.")
        
    def as_retriever(self, k: int = Config.RETRIEVAL_K):
        return self._db.as_retriever(search_kwargs={"k": k})
    
    def get_all_documents(self) -> List[LangChainDocument]:
        """Retrieves all documents from the vector store (for BM25 init)."""
        data = self._db.get()
        docs = []
        if data and data['documents']:
            for i, text in enumerate(data['documents']):
                meta = data['metadatas'][i] if data['metadatas'] else {}
                docs.append(LangChainDocument(page_content=text, metadata=meta))
        return docs
