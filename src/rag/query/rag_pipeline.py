from typing import List, Dict
import os
import redis

from src.common.config import Config
from src.rag.vector_db import VectorDatabase
from src.rag.query.retriever import HybridRetriever
from src.rag.query.reranker import DocumentReranker
from src.rag.query.prompter import PromptFormatter
from src.rag.query.llm_client import LLMClient
from src.rag.query.query_expander import QueryExpander
from src.rag.query.context_compressor import ContextCompressor
from src.rag.query.prompt_cache_manager import PromptCacheManager
from src.rag.query.answer_validator import AnswerValidator


class RAGSystem:
    def __init__(self):
        """
        Initializes the RAG system with configurable API provider (OpenAI, Anthropic, or Gemini),
        Hybrid Retrieval (BM25 + Vector), and CrossEncoder reranking.
        """
        print("Initializing RAG System...")
        
        # Initialize LLM client
        self.llm_client = LLMClient(Config.MODEL_PROVIDER)

        # Initialize vector database
        self.vector_db = VectorDatabase()

        # Initialize retriever
        self.retriever = HybridRetriever(self.vector_db)

        # Initialize reranker
        self.reranker = DocumentReranker()
        
        # Initialize prompt formatter
        self.prompter = PromptFormatter()
        
        # Initialize query expander (Phase 1 improvement)
        if Config.ENABLE_QUERY_EXPANSION:
            self.query_expander = QueryExpander(self.llm_client)
        
        # Initialize context compressor (Phase 1 improvement)
        if Config.COMPRESSION_ENABLED:
            self.context_compressor = ContextCompressor(self.reranker)
        
        # Initialize prompt cache manager (Phase 1 improvement)
        redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        try:
            redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()  # Test connection
            self.cache_manager = PromptCacheManager(redis_client)
            print("Prompt caching enabled.")
        except Exception as e:
            print(f"Warning: Could not connect to Redis for caching: {e}")
            self.cache_manager = None
        
        # Initialize answer validator (Phase 2 improvement)
        if Config.VALIDATE_ANSWERS:
            self.answer_validator = AnswerValidator(self.llm_client)
            print("Answer validation enabled.")

        print("RAG System initialized.")

    def query(
        self,
        user_query: str,
        use_expansion: bool = None,
        use_compression: bool = None,
        validate_answer: bool = None
    ) -> str:
        """
        Full RAG pipeline: Hybrid Retrieve → Rerank → Generate → Validate
        
        Args:
            user_query: User's question
            use_expansion: Override config for query expansion (None = use config)
            use_compression: Override config for context compression (None = use config)
            validate_answer: Override config for answer validation (None = use config)
            
        Returns:
            Generated answer
        """
        # Use config values if not explicitly overridden
        use_expansion = Config.ENABLE_QUERY_EXPANSION if use_expansion is None else use_expansion
        use_compression = Config.COMPRESSION_ENABLED if use_compression is None else use_compression
        validate_answer = Config.VALIDATE_ANSWERS if validate_answer is None else validate_answer
        
        print("Retrieving documents...")
        
        # Query expansion with fusion (Phase 1 improvement)
        if use_expansion and hasattr(self, 'query_expander'):
            query_variants = self.query_expander.expand_query(
                user_query,
                num_variants=Config.QUERY_EXPANSION_VARIANTS
            )
            print(f"Generated {len(query_variants)} query variants")
            retrieved_docs = self.query_expander.retrieve_with_fusion(
                query_variants,
                self.retriever
            )
        else:
            retrieved_docs = self.retriever.retrieve(user_query)

        if not retrieved_docs:
            return (
                "No relevant documents found in the database. "
                "Please ingest papers first."
            )

        reranked_docs = self.reranker.rerank(user_query, retrieved_docs)
        
        # Context compression (Phase 1 improvement)
        if use_compression and hasattr(self, 'context_compressor'):
            compressed_docs = self.context_compressor.compress_documents(
                reranked_docs,
                user_query,
                max_tokens=Config.MAX_CONTEXT_TOKENS
            )
            print(f"Compressed context to {Config.MAX_CONTEXT_TOKENS} tokens")
        else:
            compressed_docs = reranked_docs

        prompt = self.prompter.format_rag_prompt(user_query, compressed_docs)

        print(f"Generating answer with {self.llm_client.provider} ({self.llm_client.model_name})...")
        answer = self.llm_client.generate(prompt)
        
        # Answer validation (Phase 2 improvement)
        if validate_answer and hasattr(self, 'answer_validator'):
            print("Validating answer...")
            context_text = "\n\n".join([doc.page_content for doc in compressed_docs])
            validation = self.answer_validator.validate_answer(
                user_query,
                answer,
                context_text,
                threshold=Config.VALIDATION_THRESHOLD
            )
            
            if not validation["valid"]:
                print(f"Validation failed: {validation.get('issues', [])}. Improving answer...")
                answer = self.answer_validator.improve_answer(
                    user_query,
                    answer,
                    context_text,
                    validation
                )
        
        return answer

    def get_citations(self, user_query: str) -> List[Dict]:
        """
        Returns citations for a given query.
        
        Args:
            user_query: User's question
            
        Returns:
            List of citation dictionaries
        """
        retrieved_docs = self.retriever.retrieve(user_query)
        reranked_docs = self.reranker.rerank(user_query, retrieved_docs)

        citations = []
        for doc in reranked_docs:
            citations.append(
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "arxiv_id": doc.metadata.get("source", "N/A"),
                    "url": doc.metadata.get("url", "N/A"),
                    "snippet": doc.page_content[:200] + "...",
                }
            )

        return citations
