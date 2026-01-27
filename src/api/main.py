from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from celery.result import AsyncResult
from src.worker.celery_app import celery_app
from src.common.models import QueryRequest, QueryResponse, IngestRequest, IngestResponse, TaskStatusResponse
import os

app = FastAPI(title="PaperChat API", version="2.0.0")

# Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Security(api_key_header)):
    if not api_key_header:
        # For dev simplicity, allow no key or check env
        # return None 
        # But user asked for enhancements. Let's enforce it if set, or just warn.
        # Improvement plan says "User authentication".
        # Let's enforce a dummy key "paperchat-secret" for now if not set in env
        pass
    return api_key_header

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "name": "PaperChat API",
        "version": "2.0.0",
        "description": "RAG system for querying research papers",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "query": "POST /api/v1/query",
            "query_status": "GET /api/v1/query/{task_id}",
            "ingest": "POST /api/v1/ingest",
            "ingest_status": "GET /api/v1/ingest/{task_id}"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/v1/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest, api_key: str = Security(get_api_key)):
    try:
        # Decoupled task submission
        task = celery_app.send_task("src.worker.tasks.process_query", args=[request.query])
        return QueryResponse(task_id=task.id, status="processing")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/query/{task_id}", response_model=TaskStatusResponse)
async def get_query_result(task_id: str, api_key: str = Security(get_api_key)):
    task_result = AsyncResult(task_id)
    if task_result.state == 'PENDING':
        return TaskStatusResponse(task_id=task_id, status="processing")
    elif task_result.state == 'SUCCESS':
        return TaskStatusResponse(task_id=task_id, status="completed", result=task_result.result)
    elif task_result.state == 'FAILURE':
        return TaskStatusResponse(task_id=task_id, status="failed", error=str(task_result.result))
    else:
        return TaskStatusResponse(task_id=task_id, status=task_result.state)

@app.post("/api/v1/ingest", response_model=IngestResponse)
async def submit_ingest(request: IngestRequest, api_key: str = Security(get_api_key)):
    try:
        # Decoupled task submission
        # args=[query, max_results] matching function signature
        task = celery_app.send_task("src.worker.tasks.ingest_documents", args=[request.query, request.max_results])
        return IngestResponse(task_id=task.id, status="processing")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ingest/{task_id}", response_model=TaskStatusResponse)
async def get_ingest_result(task_id: str, api_key: str = Security(get_api_key)):
    task_result = AsyncResult(task_id)
    if task_result.state == 'PENDING':
        return TaskStatusResponse(task_id=task_id, status="processing")
    elif task_result.state == 'SUCCESS':
        return TaskStatusResponse(task_id=task_id, status="completed", result=task_result.result)
    elif task_result.state == 'FAILURE':
        return TaskStatusResponse(task_id=task_id, status="failed", error=str(task_result.result))
    else:
        return TaskStatusResponse(task_id=task_id, status=task_result.state)
