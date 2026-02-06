from fastapi import FastAPI, HTTPException, Depends, Security, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from celery.result import AsyncResult
from src.worker.celery_app import celery_app
from src.common.models import (
    QueryRequest, QueryResponse, IngestRequest, IngestResponse, 
    TaskStatusResponse, SuggestQuestionsRequest, SuggestQuestionsResponse
)
from src.api.auth import (
    oauth, get_current_user, require_auth, create_access_token, create_refresh_token,
    save_user_session, revoke_user_session, get_oauth_user, User,
    CALLBACK_URL_BASE, SECRET_KEY
)
import os
import json
import shutil
import logging
from datetime import timedelta, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

app = FastAPI(title="PaperChat API", version="2.0.0", docs_url='/docs')

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add SessionMiddleware for OAuth (required by authlib)
app.add_middleware(
    SessionMiddleware,
    secret_key=SECRET_KEY,
    max_age=3600  # 1 hour session for OAuth state
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the main UI"""
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# ============================================
# Authentication Endpoints
# ============================================

@app.get("/api/v1/auth/me")
async def get_current_user_info(user: User = Depends(get_current_user)):
    """Get current authenticated user info"""
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


@app.post("/api/v1/auth/logout")
async def logout(user: User = Depends(get_current_user)):
    """Logout user and revoke session"""
    if user:
        revoke_user_session(user.user_id)
    return {"message": "Logged out successfully"}


@app.post("/api/v1/auth/refresh")
async def refresh_token_endpoint(request: Request):
    """Refresh access token using refresh token"""
    try:
        body = await request.json()
        refresh_token = body.get("refresh_token")
        
        if not refresh_token:
            raise HTTPException(status_code=400, detail="Refresh token required")
        
        # Verify refresh token
        from src.api.auth import verify_token, create_access_token, REFRESH_TOKEN_EXPIRE_MINUTES, INACTIVITY_TIMEOUT_MINUTES, ALGORITHM, SECRET_KEY
        from jose import jwt, JWTError
        
        token_data = verify_token(refresh_token)
        if not token_data:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Check if token type is refresh
        try:
            payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
            if payload.get("type") != "refresh":
                raise HTTPException(status_code=401, detail="Not a refresh token")
        except JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Get user from Redis
        from src.api.auth import get_redis_client
        redis_client = get_redis_client()
        user_key = f"user:{token_data.user_id}"
        refresh_key = f"refresh:{token_data.user_id}"
        
        user_data = redis_client.get(user_key)
        stored_refresh = redis_client.get(refresh_key)
        
        if not user_data or not stored_refresh:
            raise HTTPException(status_code=401, detail="Session expired")
        
        # Verify refresh token matches stored one
        if stored_refresh != refresh_token:
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        
        # Check inactivity timeout
        from datetime import datetime
        activity_key = f"activity:{token_data.user_id}"
        last_activity = redis_client.get(activity_key)
        
        if last_activity:
            last_activity_time = datetime.fromisoformat(last_activity)
            inactive_minutes = (datetime.now(timezone.utc) - last_activity_time).total_seconds() / 60
            
            if inactive_minutes > INACTIVITY_TIMEOUT_MINUTES:
                # Session expired - force re-login
                revoke_user_session(token_data.user_id)
                raise HTTPException(
                    status_code=401,
                    detail="Session expired due to inactivity. Please login again."
                )
        
        # Create new access token
        new_access_token = create_access_token(
            data={"sub": token_data.user_id, "email": token_data.email}
        )
        
        # Update last activity
        redis_client.setex(
            activity_key,
            REFRESH_TOKEN_EXPIRE_MINUTES * 60,
            datetime.now(timezone.utc).isoformat()
        )
        
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("Token refresh failed: %s", e)
        raise HTTPException(status_code=401, detail="Token refresh failed")


@app.get("/api/v1/auth/login/{provider}")
async def login(provider: str, request: Request):
    """Initiate OAuth login flow"""
    if provider not in ['google', 'microsoft', 'github']:
        raise HTTPException(status_code=400, detail="Invalid provider")
    
    redirect_uri = f"{CALLBACK_URL_BASE}/api/v1/auth/callback/{provider}"
    return await oauth.create_client(provider).authorize_redirect(request, redirect_uri)


@app.get("/api/v1/auth/callback/{provider}")
async def auth_callback(provider: str, request: Request):
    """OAuth callback endpoint"""
    try:
        client = oauth.create_client(provider)
        token = await client.authorize_access_token(request)
        
        # Get user info from provider
        user = await get_oauth_user(provider, token)
        
        # Create JWT tokens
        access_token = create_access_token(
            data={"sub": user.user_id, "email": user.email}
        )
        refresh_token = create_refresh_token(
            data={"sub": user.user_id, "email": user.email}
        )
        
        # Save session with refresh token
        save_user_session(user, access_token, refresh_token)
        
        # Redirect to frontend with both tokens
        return RedirectResponse(
            url=f"/?access_token={access_token}&refresh_token={refresh_token}",
            status_code=302
        )
    
    except Exception as e:
        return RedirectResponse(
            url=f"/?error=auth_failed&message={str(e)}",
            status_code=302
        )


# ============================================
# Protected Endpoints (Require Authentication)
# ============================================

@app.post("/api/v1/query", response_model=QueryResponse)
async def submit_query(request: QueryRequest, user: User = Depends(require_auth)):
    """
    Submit a query (requires authentication).
    """
    try:
        # Decoupled task submission
        task = celery_app.send_task("src.worker.tasks.process_query", args=[request.query])
        return QueryResponse(task_id=task.id, status="processing")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/query/{task_id}", response_model=TaskStatusResponse)
async def get_query_result(task_id: str, user: User = Depends(require_auth)):
    """Get query result (requires authentication)"""
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
async def submit_ingest(request: IngestRequest, user: User = Depends(require_auth)):
    """Submit ingestion task (requires authentication)"""
    try:
        # Decoupled task submission
        # args=[query, max_results, skip_papers] matching function signature
        task = celery_app.send_task(
            "src.worker.tasks.ingest_documents", 
            args=[request.query, request.max_results, request.skip_papers]
        )
        return IngestResponse(task_id=task.id, status="processing")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/ingest/{task_id}", response_model=TaskStatusResponse)
async def get_ingest_result(task_id: str, user: User = Depends(require_auth)):
    """Get ingestion result (requires authentication)"""
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
async def stream_query(request: QueryRequest, user: User = Depends(require_auth)):
    """
    Stream query response for progressive delivery (requires authentication).
    
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
            
            # Send citations
            citations = rag_system.get_citations(request.query)
            yield f"data: {json.dumps({'type': 'citations', 'data': citations})}\n\n"
            
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

@app.post("/api/v1/upload", response_model=IngestResponse)
async def upload_paper(
    file: UploadFile = File(...),
    user: User = Depends(require_auth)
):
    """
    Upload a PDF paper for ingestion (requires authentication).
    
    Args:
        file: PDF file to upload
        user: Authenticated user
        
    Returns:
        Task ID for processing the uploaded paper
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are supported"
            )
        
        # Create uploads directory if it doesn't exist
        from src.common.config import Config
        uploads_dir = Path(Config.DATA_DIR) / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        file_path = uploads_dir / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Submit processing task
        task = celery_app.send_task(
            "src.worker.tasks.process_uploaded_pdf",
            args=[str(file_path), file.filename]
        )
        
        return IngestResponse(task_id=task.id, status="processing")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Upload failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/suggest-questions", response_model=SuggestQuestionsResponse)
async def suggest_questions(
    request: SuggestQuestionsRequest,
    user: User = Depends(require_auth)
):
    """
    Generate suggested follow-up questions (requires authentication).
    
    Args:
        request: Contains current query, answer, and chat history
        user: Authenticated user
        
    Returns:
        Task ID for generating suggestions
    """
    try:
        task = celery_app.send_task(
            "src.worker.tasks.generate_suggested_questions",
            args=[request.current_query, request.answer, request.chat_history or []]
        )
        
        return SuggestQuestionsResponse(task_id=task.id, status="processing")
        
    except Exception as e:
        logger.error("Question suggestion failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
