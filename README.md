# PaperChat API

PaperChat is a Retrieval-Augmented Generation service for arXiv papers and uploaded PDFs. It exposes a
FastAPI REST API, processes ingestion and query jobs with Celery, and stores chunks in a persistent ChromaDB collection.

## Architecture

```text
Client → FastAPI → Celery/Redis → RAG worker
                                  ├─ PDF parsing + recursive chunking
                                  ├─ Chroma dense retrieval + BM25
                                  ├─ Reciprocal Rank Fusion
                                  ├─ Cross-encoder reranking
                                  └─ LLM answer with source citations
```

Retrieval uses dense and lexical search in parallel. Reciprocal Rank Fusion combines their rankings without treating
incomparable search scores as equivalent. A cross-encoder reranks the fused candidates before generation. The answer
cache stores exact normalized requests and includes the corpus version and generation/retrieval settings in its key.
Ingestion advances the corpus version, so cached answers from an older index are no longer used.

Query expansion and context compression are optional RAG techniques, not required parts of a standard pipeline.
Expansion can help with ambiguous or vocabulary-mismatched queries, at the cost of extra LLM work and retrievals.
Compression can reduce context size for long retrieved passages, but adds processing and can remove useful evidence.
This service keeps the baseline deterministic and uses chunking, retrieval limits, and reranking; add either technique
only after evaluation shows a quality or cost benefit for your corpus.

## Run locally with Docker Compose

Requirements: Docker with Compose and an API key for one supported LLM provider.

```bash
cp .env.example .env
# Edit .env and set the provider API key and PAPERCHAT_API_KEY.
docker compose up --build
```

The API is available at `http://localhost:8000`; interactive OpenAPI docs are at `/docs`. Redis, the API, and the Celery
worker run as separate services. ChromaDB data and downloaded papers are persisted in the `paperchat_data` volume. The
worker runs one task at a time because this deployment uses a local persistent Chroma database; higher-throughput
deployments should use a shared vector database service and a production sparse-search backend.

For a local run without Docker, install `src/api/requirements.txt` and `src/worker/requirements.txt`, start Redis, then
run:

```bash
uvicorn src.api.main:app --reload --port 8000
celery -A src.worker.celery_app worker --loglevel=info --concurrency=1
```

Configure `CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND` to point to the same reachable Redis service. Set
`PAPERCHAT_API_KEY` for API authentication. If it is unset, authentication is disabled for local development.

## Repeatable retrieval and answer evaluation

The evaluation runner uses the same retrieval, reranking, prompt, and generation components as the query worker. It
bypasses the task queue and answer cache, so each case measures a fresh pipeline run. The starter set references
arXiv papers `1706.03762` and `1810.04805`; ingest both into the same Chroma data directory before running the
benchmark. Submit an ingestion request to `/api/v1/ingest` for each ID with `{"query":"id:1706.03762","max_results":1}`
(then repeat with `1810.04805`) and wait for both to finish. Include the configured `X-API-Key` header if API
authentication is enabled.

Install the API and worker requirements, configure a provider API key in `.env`, then run:

```bash
python -m evals.run
```

The command writes a JSON report to `evals/results/latest.json` (ignored by Git). Use `--dataset` to select another
JSONL benchmark and `--output` to preserve a separate report. Each JSONL row requires a unique `id`, a `question`,
and non-empty `relevant_source_ids`; `supporting_source_ids` can separately annotate acceptable citations, and
`reference_answer` is included for human review. Source IDs are matched without arXiv version suffixes.

The report includes pre-reranking Recall@k and MRR, recall in the final reranked context, citation precision/recall/F1
against the annotated supporting source IDs, inline citation-label validity, and mean/p50/p95 latency for retrieval,
reranking, generation, and the full uncached query. It also records answers beside reference answers for manual
correctness review. Citation scores measure agreement with the benchmark annotations; they do not by themselves
establish factual correctness. The initial retrieval warm-up and model initialization times are reported separately.
The report records dataset and corpus hashes plus the model and retrieval settings. For model comparisons, keep the
dataset, corpus, retrieval settings, provider/model, and temperature fixed; generation latency includes network and
provider time.

## API

All endpoints except `/health` require `X-API-Key` when `PAPERCHAT_API_KEY` is configured.

### Health

```http
GET /health
```

### Submit a query

```bash
curl -X POST http://localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-api-key' \
  -d '{"query":"What are the main contributions of transformers?","top_k":5,"use_cache":true}'
```

The API returns a task ID with HTTP 202. Poll `GET /api/v1/query/{task_id}` for the answer and citations:

```json
{
  "task_id": "...",
  "status": "completed",
  "result": {
    "answer": "... [1]",
    "citations": [
      {"number": 1, "title": "...", "arxiv_id": "...", "url": "...", "snippet": "..."}
    ]
  }
}
```

`filter_metadata` can restrict retrieval by exact chunk metadata values. `top_k` controls the number of reranked chunks
used to build the answer context. Set `use_cache` to `false` to bypass answer caching.

### Ingest arXiv papers

```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-api-key' \
  -d '{"query":"cat:cs.AI","max_results":10,"skip_papers":0}'
```

Poll `GET /api/v1/ingest/{task_id}` for status.

### Upload a PDF

```bash
curl -X POST http://localhost:8000/api/v1/ingest/upload \
  -H 'X-API-Key: your-api-key' \
  -F 'file=@paper.pdf'
```

The upload limit defaults to 50 MiB and can be changed with `MAX_UPLOAD_SIZE_BYTES`.

## Configuration

| Variable                                                | Purpose                                     | Default                                |
|---------------------------------------------------------|---------------------------------------------|----------------------------------------|
| `MODEL_PROVIDER`                                        | `openai`, `anthropic`, or `gemini`          | `openai`                               |
| `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY` | Provider credentials                        | unset                                  |
| `PAPERCHAT_API_KEY`                                     | REST API key; unset disables authentication | unset                                  |
| `CELERY_BROKER_URL`                                     | Celery broker                               | `redis://localhost:6379/0`             |
| `CELERY_RESULT_BACKEND`                                 | Celery result backend                       | same as broker                         |
| `EMBEDDING_MODEL_ID`                                    | Sentence embedding model                    | `BAAI/bge-large-en-v1.5`               |
| `RERANKER_MODEL_ID`                                     | Cross-encoder model                         | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP`                          | Recursive chunking size and overlap         | `1000` / `200`                         |
| `RETRIEVAL_K` / `RERANK_TOP_K`                          | Candidate and final chunk counts            | `10` / `5`                             |
| `CACHE_TTL_SECONDS`                                     | Answer cache lifetime                       | `3600`                                 |

## Project layout

```text
src/api/          FastAPI routes and request/response models
src/worker/       Celery app and background tasks
src/rag/          Ingestion, vector storage, retrieval, reranking, generation
src/common/       Configuration and API schemas
```

## License

MIT. See [LICENSE](LICENSE).
