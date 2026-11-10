---
title: PaperChat API
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: mit
tags:
  - rag
  - retrieval
  - llm
  - research-papers
  - arxiv
  - fastapi
---

# PaperChat API

PaperChat is a FastAPI backend for indexing research papers and answering questions with citations. The Docker entrypoint starts the API, Celery worker, and Redis for a single-container Space. For multi-container deployments, use the separate services in `compose.yaml`.

Set the required LLM provider credentials in Space secrets. Set `PAPERCHAT_API_KEY` to protect the API; `/health` remains public. API documentation is available at `/docs` and `/redoc`.

Endpoints:

- `GET /health`
- `POST /api/v1/query`, then poll `GET /api/v1/query/{task_id}`
- `POST /api/v1/ingest`, then poll `GET /api/v1/ingest/{task_id}`
- `POST /api/v1/ingest/upload` with a PDF multipart file

See [README.md](README.md) for request examples, configuration, and retrieval details.
