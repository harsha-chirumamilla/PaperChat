import os

class Config:
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Use /data for HuggingFace Spaces (persistent volume), /app/data otherwise
    # HF Spaces sets SPACE_ID environment variable
    if os.getenv("SPACE_ID") or os.getenv("HF_SPACE"):
        DATA_DIR = "/data"
    else:
        DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
    
    PAPERS_DIR = os.path.join(DATA_DIR, "papers")
    DB_DIR = os.path.join(DATA_DIR, "db")

    # Model Settings
    MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")  # "openai", "anthropic", or "gemini"
    
    # OpenAI Settings
    OPENAI_MODEL_ID = os.getenv("OPENAI_MODEL_ID", "gpt-4-turbo")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Anthropic Settings
    ANTHROPIC_MODEL_ID = os.getenv("ANTHROPIC_MODEL_ID", "claude-3-5-sonnet-20241022")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Google Gemini Settings
    GEMINI_MODEL_ID = os.getenv("GEMINI_MODEL_ID", "gemini-2.0-flash-exp")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    
    # Embedding and Reranker Models
    EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-large-en-v1.5")
    RERANKER_MODEL_ID = os.getenv("RERANKER_MODEL_ID", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # RAG Parameters
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 10))
    RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))

    # Generation Parameters
    MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 512))
    TEMPERATURE = float(os.getenv("TEMPERATURE", 0.3))

    # Ingestion
    MAX_PAPERS_PER_BATCH = int(os.getenv("MAX_PAPERS_PER_BATCH", 5))
    
    # Query Expansion
    ENABLE_QUERY_EXPANSION = os.getenv("ENABLE_QUERY_EXPANSION", "true").lower() == "true"
    QUERY_EXPANSION_VARIANTS = int(os.getenv("QUERY_EXPANSION_VARIANTS", 2))
    
    # Context Compression
    MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 2500))
    COMPRESSION_ENABLED = os.getenv("COMPRESSION_ENABLED", "true").lower() == "true"
    
    # Agentic Workflow (expensive - disabled by default)
    ENABLE_AGENT = os.getenv("ENABLE_AGENT", "false").lower() == "true"
    AGENT_MAX_ITERATIONS = int(os.getenv("AGENT_MAX_ITERATIONS", 5))
    
    # Answer Validation (expensive - disabled by default for cost efficiency)
    VALIDATE_ANSWERS = os.getenv("VALIDATE_ANSWERS", "false").lower() == "true"
    VALIDATION_THRESHOLD = float(os.getenv("VALIDATION_THRESHOLD", 0.7))
    
    # Hierarchical Chunking
    USE_HIERARCHICAL_CHUNKING = os.getenv("USE_HIERARCHICAL_CHUNKING", "false").lower() == "true"

# Create directories if they don't exist
os.makedirs(Config.PAPERS_DIR, exist_ok=True)
os.makedirs(Config.DB_DIR, exist_ok=True)
