# RAG Pipeline for arXiv Papers

A scalable, API-based Retrieval-Augmented Generation (RAG) system that indexes computer science and AI papers from arXiv and answers questions using configurable LLM providers (OpenAI, Anthropic, or Gemini).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │    FastAPI Server     │
                │   Celery Task Queue   │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Hybrid Retrieval    │
                │  BM25 + Vector Search │
                │  (all-MiniLM-L6-v2)   │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   CrossEncoder        │
                │    Reranking (Top-3)  │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Prompt Formatting   │
                │  Context + Question   │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   LLM API Client      │
                │ OpenAI/Anthropic/     │
                │      Gemini           │
                └───────────┬───────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   Grounded Answer     │
                │    + Citations        │
                └───────────────────────┘
```

## ✨ Features

- **Multi-Provider LLM Support**: OpenAI, Anthropic, or Gemini via API
- **GPU-Accelerated Embeddings**: Optimized for CUDA-enabled devices
- **Hybrid Retrieval**: BM25 + Vector search with CrossEncoder reranking
- **arXiv Integration**: Automatic paper download and processing
- **Async Task Processing**: Celery workers with Redis caching
- **REST API**: FastAPI server for scalable deployment
- **Docker Support**: Reproducible containerized environment

## 🧰 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| LLM APIs | OpenAI/Anthropic/Gemini | Answer generation |
| Embeddings | all-MiniLM-L6-v2 | Semantic search |
| Reranker | CrossEncoder | Result refinement |
| Retrieval | BM25 + Vector Search | Hybrid document retrieval |
| Vector DB | ChromaDB | Persistent document storage |
| Task Queue | Celery + Redis | Async processing & caching |
| API Server | FastAPI | REST endpoints |
| PDF Parser | PyMuPDF | Text extraction |
| Container | Docker | Isolated environment |

## 📋 Prerequisites

- **API Keys**: At least one of:
  - OpenAI API key ([platform.openai.com](https://platform.openai.com))
  - Anthropic API key ([console.anthropic.com](https://console.anthropic.com))
  - Google Gemini API key ([aistudio.google.com](https://aistudio.google.com))
- **Hardware** (Optional): NVIDIA GPU for faster embeddings/reranking
- **Software**:
  - Docker Desktop (Windows/Mac) or Docker Engine (Linux)
  - NVIDIA Container Toolkit (optional, for GPU acceleration)

## 🚀 Quick Start

### 1. Configure Environment

Create a `.env` file in the project root:

```bash
# Choose one provider: openai, anthropic, or gemini
MODEL_PROVIDER=gemini

# Add your API key (only one required)
GEMINI_API_KEY=your_gemini_api_key_here
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here

# Celery configuration
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0
```

### 2. Build and Start Services

```bash
docker-compose build
docker-compose up -d
```

This starts:
- Redis (caching & task queue)
- Celery Worker (document processing & queries)
- FastAPI Server (REST API on port 8000)

### 3. Ingest arXiv Papers

Using the CLI:
```bash
docker-compose exec worker python src/ingestion_entrypoint.py \
  --category "cat:cs.AI" \
  --max-papers 10
```

Or via API:
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"query": "cat:cs.AI", "max_results": 10}'
```

**Popular categories:**
- `cat:cs.LG` - Machine Learning
- `cat:cs.CL` - Computation and Language (NLP)
- `cat:cs.CV` - Computer Vision
- `cat:cs.AI` - Artificial Intelligence

### 4. Query the System

Using the CLI:
```bash
docker-compose exec worker python src/query.py \
  --question "What are transformer attention mechanisms?"
```

Or via API:
```bash
# Submit query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are transformer attention mechanisms?"}'

# Get results (replace {task_id} with response from above)
curl http://localhost:8000/api/v1/query/{task_id}
```

## 🔧 Configuration

Edit `src/common/config.py` or set environment variables:

```python
class Config:
    # LLM Provider (set via MODEL_PROVIDER env var)
    MODEL_PROVIDER = "gemini"  # openai, anthropic, or gemini
    
    # Model IDs per provider
    OPENAI_MODEL_ID = "gpt-4o"
    ANTHROPIC_MODEL_ID = "claude-3-5-sonnet-20241022"
    GEMINI_MODEL_ID = "gemini-2.0-flash-exp"
    
    # Embedding & Reranking
    EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
    RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    # Retrieval
    RETRIEVAL_K = 10           # Initial retrieval count
    RERANK_TOP_K = 3           # Top docs after reranking
    
    # Generation
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.7
    
    # Chunking
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
```

## 🎯 API Endpoints

### Health Check
```bash
GET /health
```

### Submit Query
```bash
POST /api/v1/query
{
  "query": "What are transformer architectures?"
}

Response:
{
  "task_id": "abc-123",
  "status": "processing"
}
```

### Get Query Results
```bash
GET /api/v1/query/{task_id}

Response:
{
  "task_id": "abc-123",
  "status": "completed",
  "result": {
    "answer": "Transformers are neural network architectures...",
    "citations": [
      {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "url": "https://arxiv.org/abs/1706.03762",
        "snippet": "..."
      }
    ]
  }
}
```

### Submit Ingestion
```bash
POST /api/v1/ingest
{
  "query": "cat:cs.AI",
  "max_results": 10
}
```

### Get Ingestion Status
```bash
GET /api/v1/ingest/{task_id}
```

## 🐳 Docker Architecture

### Services

**redis**: Task queue broker and result cache
**worker**: Celery worker for async task processing
**api**: FastAPI server (optional, for REST API)

### GPU Support (Optional)

If you have an NVIDIA GPU, embeddings and reranking will automatically use CUDA:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

CPU-only mode works fine but will be slower for embeddings.

### Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `./data/papers` | `/app/data/papers` | Downloaded PDFs |
| `./data/db` | `/app/data/db` | ChromaDB vector store |
| `./data/huggingface` | `/app/venv/lib/.../site-packages/.../hub` | Model cache |
| `./src` | `/app/src` | Source code (live updates) |

## ⚡ Performance Optimization

### GPU Memory (if using CUDA)
- Embedding Model: ~0.5GB VRAM
- CrossEncoder Reranker: ~0.3GB VRAM
- **Total VRAM needed: ~1GB** (works on most GPUs)

### API Cost Optimization

1. **Use Redis caching** (enabled by default):
   - Identical queries return cached results
   - 1-hour cache expiration

2. **Reduce context tokens**:
   ```python
   RERANK_TOP_K = 2  # Fewer documents in context
   MAX_NEW_TOKENS = 256  # Shorter responses
   ```

3. **Choose cost-effective models**:
   ```bash
   # Gemini is typically cheapest
   MODEL_PROVIDER=gemini
   GEMINI_MODEL_ID=gemini-2.0-flash-exp
   ```

4. **Use smaller embedding model**:
   ```python
   EMBEDDING_MODEL_ID = "sentence-transformers/paraphrase-MiniLM-L3-v2"
   ```

## 🛠️ Troubleshooting

### "API key not set"
```bash
# Check your .env file has the correct key
MODEL_PROVIDER=gemini
GEMINI_API_KEY=your_actual_key_here

# Restart services
docker-compose down
docker-compose up -d
```

### "No relevant documents found"
```bash
# Verify documents were ingested
docker-compose exec worker python -c "
from src.rag.vector_db import VectorDatabase
db = VectorDatabase()
print(f'Documents in DB: {len(db.get_all_documents())}')
"

# If empty, run ingestion first
docker-compose exec worker python src/ingestion_entrypoint.py \
  --category "cat:cs.AI" --max-papers 10
```

### Connection errors
```bash
# Ensure all services are running
docker-compose ps

# Should see: redis, worker (and optionally api)
# If not, check logs:
docker-compose logs redis
docker-compose logs worker
```

### GPU not detected (optional)
```bash
# Check NVIDIA toolkit installation
docker run --rm --gpus all nvidia/cuda:13.1.1-runtime-ubuntu24.04 nvidia-smi

# If fails, the system will fall back to CPU (slower but still works)
```
