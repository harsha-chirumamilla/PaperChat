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


class RAGSystem:
    def __init__(self):
        """
        Initializes the RAG system with configurable API provider (OpenAI, Anthropic, or Gemini),
        Hybrid Retrieval (BM25 + Vector), and CrossEncoder reranking.
        """
        logger.info("Initializing RAG System")
        
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
        
        logger.info("Retrieving documents for query")
        
        # Query expansion with fusion (Phase 1 improvement)
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
            logger.info("Compressed context to %d tokens", Config.MAX_CONTEXT_TOKENS)
        else:
            compressed_docs = reranked_docs

        prompt = self.prompter.format_rag_prompt(user_query, compressed_docs)

        logger.info("Generating answer with %s (%s)", self.llm_client.provider, self.llm_client.model_name)
        answer = self.llm_client.generate(prompt)
        
        # Answer validation (Phase 2 improvement)
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
