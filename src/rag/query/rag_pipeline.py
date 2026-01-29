from typing import List, Dict, Optional
import os
import logging
import redis

from src.common.config import Config

logger = logging.getLogger(__name__)
from src.rag.vector_db import VectorDatabase
from src.rag.query.retriever import HybridRetriever
from src.rag.query.reranker import DocumentReranker
from src.rag.query.prompter import PromptFormatter
from src.rag.query.llm_client import LLMClient
from src.rag.query.query_expander import QueryExpander
from src.rag.query.context_compressor import ContextCompressor
from src.rag.query.prompt_cache_manager import PromptCacheManager
from src.rag.query.answer_validator import AnswerValidator
from src.rag.query.query_classifier import QueryComplexityClassifier
from src.rag.query.semantic_cache import SemanticCache
from src.rag.embeddings import get_embedding_function


class RAGSystem:
    def __init__(self):
        """
        Initializes the RAG system with configurable API provider (OpenAI, Anthropic, or Gemini),
        Hybrid Retrieval (BM25 + Vector), and CrossEncoder reranking.
        """
        logger.info("Initializing RAG System")
        
        # Performance: Cache last query results to avoid re-retrieval
        self._last_query = None
        self._last_results = None
        
        # Initialize LLM client
        self.llm_client = LLMClient(Config.MODEL_PROVIDER)

        # Initialize vector database
        self.vector_db = VectorDatabase()

        # Initialize Redis client for caching
        redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
        try:
            redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
            redis_client.ping()  # Test connection
            logger.info("Redis client initialized for retrieval caching")
        except Exception as e:
            logger.warning("Could not connect to Redis for caching: %s", e)
            redis_client = None

        # Initialize retriever with Redis cache
        self.retriever = HybridRetriever(self.vector_db, redis_client=redis_client)

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
            logger.info("Prompt caching enabled")
        except Exception as e:
            logger.warning("Could not connect to Redis for caching: %s", e)
            self.cache_manager = None
        
        # Initialize answer validator (Phase 2 improvement)
        if Config.VALIDATE_ANSWERS:
            self.answer_validator = AnswerValidator(self.llm_client)
            logger.info("Answer validation enabled")
        
        # Initialize ReAct agent (Phase 3 improvement)
        if Config.ENABLE_AGENT:
            from src.rag.agent.react_agent import ReActAgent
            self.agent = ReActAgent(self.llm_client, self)
            logger.info("ReAct agent enabled")
        
        # Initialize query classifier for smart routing
        self.query_classifier = QueryComplexityClassifier(self.llm_client)
        
        # Initialize semantic cache (Tier 3 optimization)
        try:
            redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
            redis_client_semantic = redis.Redis.from_url(redis_url, decode_responses=True)
            redis_client_semantic.ping()
            embedding_fn = get_embedding_function()
            self.semantic_cache = SemanticCache(redis_client_semantic, embedding_fn, threshold=0.92)
            logger.info("Semantic cache enabled")
        except Exception as e:
            logger.warning("Could not initialize semantic cache: %s", e)
            self.semantic_cache = None

        logger.info("RAG System initialized successfully")

    def query(
        self,
        user_query: str,
        use_expansion: bool = None,
        use_compression: bool = None,
        validate_answer: bool = None,
        use_agent: bool = None,
        filter_metadata: Dict = None,
        auto_classify: bool = True
    ) -> str:
        """
        Full RAG pipeline: Hybrid Retrieve → Rerank → Generate → Validate
        
        Args:
            user_query: User's question
            use_expansion: Override config for query expansion (None = use config)
            use_compression: Override config for context compression (None = use config)
            validate_answer: Override config for answer validation (None = use config)
            use_agent: Override config for agentic workflow (None = use config)
            filter_metadata: Optional metadata filters (e.g., {'section': 'methods'})
            auto_classify: Automatically classify query complexity and optimize token usage
            
        Returns:
            Generated answer
        """
        # Check semantic cache first (Tier 3 optimization)
        if self.semantic_cache:
            cached_result = self.semantic_cache.get(user_query)
            if cached_result:
                logger.info("Semantic cache hit for query: %s", user_query[:50])
                # Store in instance cache for get_citations()
                self._last_query = user_query
                self._last_results = {
                    'retrieved': [],
                    'reranked': [],
                    'compressed': []
                }
                return cached_result["answer"]
        
        # Auto-classify query complexity for optimal token usage
        if auto_classify and all(x is None for x in [use_expansion, use_compression, validate_answer, use_agent]):
            classification = self.query_classifier.classify(user_query)
            logger.info("Query classified as: %s (confidence: %.2f)", classification['level'], classification['confidence'])
            logger.info("Strategy: %s", classification['reasoning'])
            
            # Use classification recommendations
            use_expansion = classification['use_expansion']
            use_compression = True  # Always beneficial
            validate_answer = classification['use_validation']
            use_agent = classification['use_agent']
            
            # Estimate token usage
            estimated_tokens = self._estimate_token_usage(
                use_expansion, use_compression, validate_answer, use_agent
            )
            logger.info("Estimated tokens: ~%d", estimated_tokens)
        else:
            # Use config values if not explicitly overridden
            use_expansion = Config.ENABLE_QUERY_EXPANSION if use_expansion is None else use_expansion
            use_compression = Config.COMPRESSION_ENABLED if use_compression is None else use_compression
            validate_answer = Config.VALIDATE_ANSWERS if validate_answer is None else validate_answer
            use_agent = Config.ENABLE_AGENT if use_agent is None else use_agent
        
        # Use ReAct agent for complex queries (Phase 3 improvement)
        if use_agent and hasattr(self, 'agent'):
            logger.info("Using ReAct agent for complex query")
            result = self.agent.run(user_query)
            return result["answer"]
        
        import time
        timings = {}
        query_start = time.time()
        
        logger.info("Retrieving documents for query")
        
        # Query expansion with fusion (Phase 1 improvement)
        t0 = time.time()
        if use_expansion and hasattr(self, 'query_expander'):
            query_variants = self.query_expander.expand_query(
                user_query,
                num_variants=Config.QUERY_EXPANSION_VARIANTS
            )
            logger.info("Generated %d query variants", len(query_variants))
            retrieved_docs = self.query_expander.retrieve_with_fusion(
                query_variants,
                self.retriever
            )
        else:
            # Support metadata filtering (Phase 4 improvement)
            retrieved_docs = self.retriever.retrieve(user_query, filter_metadata=filter_metadata)
        timings['retrieval'] = time.time() - t0

        if not retrieved_docs:
            return (
                "No relevant documents found in the database. "
                "Please ingest papers first."
            )

        t0 = time.time()
        reranked_docs = self.reranker.rerank(user_query, retrieved_docs)
        timings['reranking'] = time.time() - t0
        
        # Context compression (Phase 1 improvement)
        t0 = time.time()
        if use_compression and hasattr(self, 'context_compressor'):
            compressed_docs = self.context_compressor.compress_documents(
                reranked_docs,
                user_query,
                max_tokens=Config.MAX_CONTEXT_TOKENS
            )
            logger.info("Compressed context to %d tokens", Config.MAX_CONTEXT_TOKENS)
        else:
            compressed_docs = reranked_docs
        timings['compression'] = time.time() - t0

        # Cache results for get_citations() to avoid re-retrieval
        self._last_query = user_query
        self._last_results = {
            'retrieved': retrieved_docs,
            'reranked': reranked_docs,
            'compressed': compressed_docs
        }

        prompt = self.prompter.format_rag_prompt(user_query, compressed_docs)

        t0 = time.time()
        logger.info("Generating answer with %s (%s)", self.llm_client.provider, self.llm_client.model_name)
        answer = self.llm_client.generate(prompt)
        timings['generation'] = time.time() - t0
        
        # Answer validation (Phase 2 improvement)
        t0 = time.time()
        if validate_answer and hasattr(self, 'answer_validator'):
            logger.info("Validating answer")
            context_text = "\n\n".join([doc.page_content for doc in compressed_docs])
            validation = self.answer_validator.validate_answer(
                user_query,
                answer,
                context_text,
                threshold=Config.VALIDATION_THRESHOLD
            )
            
            if not validation["valid"]:
                issues = validation.get('issues', [])
                logger.warning("Validation failed: %s. Improving answer", issues)
                answer = self.answer_validator.improve_answer(
                    user_query,
                    answer,
                    context_text,
                    validation
                )
        timings['validation'] = time.time() - t0 if validate_answer else 0
        
        timings['total'] = time.time() - query_start
        logger.info("Query completed in %.2fs - Retrieval: %.2fs, Reranking: %.2fs, Compression: %.2fs, Generation: %.2fs, Validation: %.2fs",
                   timings['total'], timings['retrieval'], timings['reranking'],
                   timings['compression'], timings['generation'], timings['validation'])
        
        # Cache result in semantic cache (Tier 3 optimization)
        if self.semantic_cache:
            citations = self.get_citations(user_query)
            self.semantic_cache.set(user_query, answer, citations)
        
        return answer
    
    def _estimate_token_usage(
        self,
        use_expansion: bool,
        use_compression: bool,
        validate_answer: bool,
        use_agent: bool
    ) -> int:
        """
        Estimate total token usage for a query with given settings.
        
        Args:
            use_expansion: Whether query expansion is enabled
            use_compression: Whether context compression is enabled
            validate_answer: Whether answer validation is enabled
            use_agent: Whether ReAct agent is enabled
            
        Returns:
            Estimated total tokens
        """
        tokens = 0
        
        # Base retrieval + generation
        if use_compression:
            tokens += Config.MAX_CONTEXT_TOKENS  # Compressed context
        else:
            tokens += Config.RERANK_TOP_K * 500  # ~500 tokens per doc
        
        tokens += 100  # Query
        tokens += Config.MAX_NEW_TOKENS  # Answer generation
        
        # Additional features
        if use_expansion:
            tokens += 150  # Query variant generation
        
        if validate_answer:
            tokens += 400  # Validation prompt
            tokens += 600  # Potential regeneration (50% chance)
        
        if use_agent:
            # ReAct is expensive!
            tokens += 3000  # Average: 3-5 iterations with thoughts/actions
        
        return tokens

    def get_citations(self, user_query: str) -> List[Dict]:
        """
        Returns citations for a given query.
        Uses cached results if available to avoid re-retrieval.
        
        Args:
            user_query: User's question
            
        Returns:
            List of citation dictionaries
        """
        # Use cached results if available (saves 1-2s)
        if self._last_query == user_query and self._last_results:
            logger.info("Using cached results for citations")
            reranked_docs = self._last_results['reranked']
        else:
            # Fallback to re-retrieval if cache miss
            logger.info("Cache miss, re-retrieving for citations")
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
