import fitz
import logging
from typing import List, Dict
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document as LangChainDocument
from src.common.config import Config
from src.rag.embeddings import get_embedding_function
from src.rag.ingestion.hierarchical_chunker import HierarchicalChunker

logger = logging.getLogger(__name__)


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
            logger.error("Error reading %s: %s", filepath, e)
            return ""
        return text

    @staticmethod
    def create_chunks(
        text: str,
        metadata: Dict,
        use_hierarchical: bool = None
    ) -> List[LangChainDocument]:
        """
        Splits text into chunks using hierarchical or semantic chunking.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            use_hierarchical: If True, use hierarchical chunker. If None, use config value
        """
        # Check config if not explicitly specified
        if use_hierarchical is None:
            use_hierarchical = Config.USE_HIERARCHICAL_CHUNKING
        
        source_id = metadata.get("id") or metadata.get("arxiv_id") or metadata.get("source") or "unknown"
        title = metadata.get("title", "Untitled")
        url = metadata.get("url", "")

        # Use hierarchical chunking for structured documents
        if use_hierarchical:
            chunker = HierarchicalChunker()
            base_meta = {
                "source": source_id,
                "title": title,
                "url": url
            }
            return chunker.create_hierarchical_chunks(
                text,
                base_meta,
                base_chunk_size=Config.CHUNK_SIZE
            )
        
        # Standard semantic chunking
        embedding_function = get_embedding_function()
        
        splitter = SemanticChunker(
            embeddings=embedding_function,
            breakpoint_threshold_type="percentile"
        )
        
        try:
            texts = splitter.split_text(text)
        except Exception as e:
            logger.warning("Semantic chunking failed: %s. Falling back to basic splitting", e)
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
                "source": source_id,
                "title": title,
                "chunk_id": i,
                "url": url
            }
            chunks.append(LangChainDocument(page_content=t, metadata=clean_meta))
            
        return chunks
