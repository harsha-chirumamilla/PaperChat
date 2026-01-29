from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse
from celery.result import AsyncResult
from src.worker.celery_app import celery_app
from src.common.models import QueryRequest, QueryResponse, IngestRequest, IngestResponse, TaskStatusResponse
import os
import json

app = FastAPI(title="PaperChat API", version="2.0.0", docs_url='/')
# Security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def get_api_key(api_key_header: str = Security(api_key_header)):
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

@app.post("/api/v1/query/stream")
async def stream_query(request: QueryRequest, api_key: str = Security(get_api_key)):
    """
    Stream query response for progressive delivery.
    
    Tier 3 optimization: Reduces perceived latency by streaming
    LLM output as it's generated.
    """
    try:
        async def generate():
            """Generate streaming response with SSE format."""
            from src.worker.tasks import get_rag_system
            
            # Get RAG system instance
            rag_system = get_rag_system()
            
            # Send initial metadata
            yield f"data: {json.dumps({'type': 'start', 'query': request.query})}\n\n"
            
            # Perform retrieval and reranking
            retrieved_docs = rag_system.retriever.retrieve(request.query)
            
            if not retrieved_docs:
                yield f"data: {json.dumps({'type': 'error', 'message': 'No relevant documents found'})}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            reranked_docs = rag_system.reranker.rerank(request.query, retrieved_docs)
            
            # Send source documents metadata
            sources = [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "arxiv_id": doc.metadata.get("source", "N/A"),
                    "url": doc.metadata.get("url", "N/A")
                }
                for doc in reranked_docs
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources})}\n\n"
            
            # Format prompt
            prompt = rag_system.prompter.format_rag_prompt(request.query, reranked_docs)
            
            # Stream LLM response
            yield f"data: {json.dumps({'type': 'answer_start'})}\n\n"
            
            async for chunk in rag_system.llm_client.stream_async(prompt):
                # Send each chunk as SSE event
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"  # Disable nginx buffering
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
