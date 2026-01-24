import fitz
from typing import List, Dict
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document as LangChainDocument
from src.common.config import Config
from src.rag.embeddings import get_embedding_function


class PDFProcessor:
    @staticmethod
    def extract_text(filepath: str) -> str:
        """Extracts text from a PDF file efficiently using PyMuPDF."""
        text = ""
        try:
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return ""
        return text

    @staticmethod
    def create_chunks(text: str, metadata: Dict) -> List[LangChainDocument]:
        """
        Splits text into chunks using SemanticChunker with cached embeddings.
        """
        # Reuse cached embedding function (loaded only once)
        embedding_function = get_embedding_function()
        
        splitter = SemanticChunker(
            embeddings=embedding_function,
            breakpoint_threshold_type="percentile"
        )
        
        try:
            texts = splitter.split_text(text)
        except Exception as e:
            print(f"Semantic chunking failed: {e}. Falling back to basic splitting.")
            # Fallback (import here to avoid circular or overhead if rarely used)
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE, 
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            texts = splitter.split_text(text)

        chunks = []
        for i, t in enumerate(texts):
            # Clean up keys for metadata to ensure compatibility
            clean_meta = {
                "source": metadata["id"],
                "title": metadata["title"],
                "chunk_id": i,
                "url": metadata["url"]
            }
            chunks.append(LangChainDocument(page_content=t, metadata=clean_meta))
            
        return chunks
