"""FastAPI endpoints for PaperChat's query and ingestion services."""

import hmac
import logging
from pathlib import Path
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.security import APIKeyHeader

from src.common.config import Config
from src.common.models import IngestRequest, IngestResponse, QueryRequest, QueryResponse, TaskStatusResponse
from src.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

app = FastAPI(
    title="PaperChat API",
    description="Backend API for indexing research papers and answering questions with citations.",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str = Depends(api_key_header)) -> None:
    """Check the configured API key; an unset key keeps local development frictionless."""
    expected = Config.API_KEY
    if expected and (not api_key or not hmac.compare_digest(api_key, expected)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )


@app.get("/health", tags=["operations"])
async def health_check():
    return {"status": "healthy"}


def _task_status(task_id: str, kind: str) -> TaskStatusResponse:
    result = AsyncResult(task_id, app=celery_app)
    if result.state in {"PENDING", "RECEIVED", "STARTED", "RETRY"}:
        return TaskStatusResponse(task_id=task_id, status="processing")
    if result.state == "SUCCESS":
        return TaskStatusResponse(task_id=task_id, status="completed", result=result.result)
    if result.state == "FAILURE":
        logger.error("%s task %s failed: %s", kind, task_id, result.result)
        return TaskStatusResponse(task_id=task_id, status="failed", error="Task failed; see service logs")
    return TaskStatusResponse(task_id=task_id, status=result.state.lower())


def _enqueue(task_name: str, args: list) -> str:
    try:
        return celery_app.send_task(task_name, args=args).id
    except Exception as exc:
        logger.exception("Could not enqueue %s", task_name)
        raise HTTPException(status_code=503, detail="Background task service unavailable") from exc


@app.post(
    "/api/v1/query",
    response_model=QueryResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["query"],
    dependencies=[Depends(require_api_key)],
)
async def submit_query(request: QueryRequest):
    task_id = _enqueue(
        "src.worker.tasks.process_query",
        [request.query, request.top_k, request.use_cache, request.filter_metadata],
    )
    return QueryResponse(task_id=task_id, status="processing")


@app.get(
    "/api/v1/query/{task_id}",
    response_model=TaskStatusResponse,
    tags=["query"],
    dependencies=[Depends(require_api_key)],
)
async def get_query_result(task_id: str):
    return _task_status(task_id, "Query")


@app.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["ingestion"],
    dependencies=[Depends(require_api_key)],
)
async def submit_ingest(request: IngestRequest):
    task_id = _enqueue(
        "src.worker.tasks.ingest_documents",
        [request.query, request.max_results, request.skip_papers],
    )
    return IngestResponse(task_id=task_id, status="processing")


@app.get(
    "/api/v1/ingest/{task_id}",
    response_model=TaskStatusResponse,
    tags=["ingestion"],
    dependencies=[Depends(require_api_key)],
)
async def get_ingest_result(task_id: str):
    return _task_status(task_id, "Ingestion")


@app.post(
    "/api/v1/ingest/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["ingestion"],
    dependencies=[Depends(require_api_key)],
)
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    uploads_dir = Path(Config.DATA_DIR) / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name
    file_path = uploads_dir / f"{uuid4().hex}.pdf"
    size = 0
    try:
        with file_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > Config.MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds the upload size limit")
                destination.write(chunk)
        task_id = _enqueue(
            "src.worker.tasks.process_uploaded_pdf",
            [str(file_path), safe_name],
        )
        return IngestResponse(task_id=task_id, status="processing")
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception:
        file_path.unlink(missing_ok=True)
        logger.exception("PDF upload failed")
        raise HTTPException(status_code=500, detail="PDF upload failed")
    finally:
        await file.close()
