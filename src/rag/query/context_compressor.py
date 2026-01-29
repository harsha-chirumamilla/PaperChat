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
        max_tokens: int = 3000,
        use_batch_scoring: bool = True
    ) -> List[Document]:
        """
        Compress documents to fit within token budget.
        
        Tier 2 optimization: Use batch scoring to combine document-level and 
        sentence-level scoring in a single GPU call.
        
        Args:
            documents: List of documents to compress
            query: User query for relevance scoring
            max_tokens: Target token budget
            use_batch_scoring: Use batch CrossEncoder scoring (faster)
            
        Returns:
            Compressed documents
        """
        if use_batch_scoring:
            return self._compress_with_batch_scoring(documents, query, max_tokens)
        
        # Fallback to sequential scoring
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
    
    def _compress_with_batch_scoring(
        self,
        documents: List[Document],
        query: str,
        max_tokens: int
    ) -> List[Document]:
        """
        Compress using batch scoring for better GPU utilization.
        
        This combines all document and sentence scoring into a single batch,
        reducing GPU overhead from multiple small calls.
        """
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Prepare batch: collect all texts (documents + sentences) to score
        batch_texts = []
        batch_metadata = []  # Track which text belongs to which document
        
        for doc_idx, doc in enumerate(documents):
            doc_tokens = len(doc.page_content) // 4
            
            # Add document itself
            batch_texts.append(doc.page_content)
            batch_metadata.append({
                'type': 'document',
                'doc_idx': doc_idx,
                'tokens': doc_tokens
            })
            
            # If document might need compression, add its sentences too
            if doc_tokens > 500:  # Threshold for compression
                sentences = nltk.sent_tokenize(doc.page_content)
                if len(sentences) > 3:
                    for sent_idx, sent in enumerate(sentences):
                        batch_texts.append(sent)
                        batch_metadata.append({
                            'type': 'sentence',
                            'doc_idx': doc_idx,
                            'sent_idx': sent_idx,
                            'text': sent
                        })
        
        # Single batch scoring call
        if batch_texts:
            scores = self.reranker.batch_score(query, batch_texts)
        else:
            scores = []
        
        # Process results
        compressed = []
        current_tokens = 0
        doc_sentence_scores = {}  # Store sentence scores per document
        
        for i, (text, metadata) in enumerate(zip(batch_texts, batch_metadata)):
            if metadata['type'] == 'sentence':
                doc_idx = metadata['doc_idx']
                if doc_idx not in doc_sentence_scores:
                    doc_sentence_scores[doc_idx] = []
                doc_sentence_scores[doc_idx].append((metadata['text'], scores[i]))
        
        # Build compressed documents
        for doc_idx, doc in enumerate(documents):
            doc_tokens = len(doc.page_content) // 4
            
            if current_tokens + doc_tokens > max_tokens:
                # Compress using pre-scored sentences
                if doc_idx in doc_sentence_scores:
                    # Get top 2 sentences by score
                    top_sentences = sorted(
                        doc_sentence_scores[doc_idx],
                        key=lambda x: x[1],
                        reverse=True
                    )[:2]
                    
                    # Restore original order
                    sentences = nltk.sent_tokenize(doc.page_content)
                    selected = sorted(
                        top_sentences,
                        key=lambda x: sentences.index(x[0]) if x[0] in sentences else 999
                    )
                    
                    compressed_text = " ".join([sent for sent, _ in selected])
                    doc.page_content = compressed_text
                    current_tokens += len(compressed_text) // 4
                else:
                    # Fallback: use full document if no sentences scored
                    current_tokens += doc_tokens
            else:
                current_tokens += doc_tokens
            
            compressed.append(doc)
            
            if current_tokens >= max_tokens:
                break
        
        return compressed
