import fitz
import logging
from typing import List, Dict
from langchain_core.documents import Document as LangChainDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.common.config import Config
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
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            add_start_index=True,
        )
        split_documents = splitter.create_documents([text])

        chunks = []
        for i, split_doc in enumerate(split_documents):
            # Clean up keys for metadata to ensure compatibility
            clean_meta = {
                "source": source_id,
                "title": title,
                "chunk_id": i,
                "start_index": split_doc.metadata.get("start_index", 0),
                "url": url
            }
            chunks.append(LangChainDocument(page_content=split_doc.page_content, metadata=clean_meta))
            
        return chunks
