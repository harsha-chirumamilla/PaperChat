from typing import List
from langchain_core.documents import Document
from src.rag.query.reranker import DocumentReranker
from src.common.config import Config


class ContextCompressor:
    """Intelligently compress context to reduce token usage."""
    
    def __init__(self, reranker: DocumentReranker):
        self.reranker = reranker
    
    def extract_key_sentences(self, text: str, query: str, max_sentences: int = 3) -> str:
        """
        Extract most relevant sentences from a chunk.
        
        Args:
            text: Document text
            query: User query
            max_sentences: Maximum sentences to extract
            
        Returns:
            Compressed text with key sentences
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        sentences = nltk.sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Score each sentence using reranker
        sentence_pairs = [[query, sent] for sent in sentences]
        scores = self.reranker.reranker.predict(sentence_pairs)
        
        # Get top sentences
        ranked_sentences = sorted(
            zip(sentences, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:max_sentences]
        
        # Return in original order
        selected = sorted(
            ranked_sentences,
            key=lambda x: sentences.index(x[0])
        )
        
        return " ".join([sent for sent, _ in selected])
    
    def compress_documents(
        self, 
        documents: List[Document], 
        query: str,
        max_tokens: int = 3000
    ) -> List[Document]:
        """
        Compress documents to fit within token budget.
        
        Args:
            documents: List of documents to compress
            query: User query for relevance scoring
            max_tokens: Target token budget
            
        Returns:
            Compressed documents
        """
        compressed = []
        current_tokens = 0
        
        for doc in documents:
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            doc_tokens = len(doc.page_content) // 4
            
            if current_tokens + doc_tokens > max_tokens:
                # Compress this document
                compressed_text = self.extract_key_sentences(
                    doc.page_content,
                    query,
                    max_sentences=2
                )
                doc.page_content = compressed_text
                current_tokens += len(compressed_text) // 4
            else:
                current_tokens += doc_tokens
            
            compressed.append(doc)
            
            if current_tokens >= max_tokens:
                break
        
        return compressed
