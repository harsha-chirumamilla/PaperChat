from pydantic import BaseModel
from typing import List, Optional, Any

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_cache: bool = True

class QueryResponse(BaseModel):
    task_id: str
    status: str

class IngestRequest(BaseModel):
    # For now, simple ingest from Arxiv query or similar
    # Or maybe list of arxiv IDs?
    # Improvement plan mentions "POST /api/v1/ingest/arxiv - Batch download"
    arxiv_ids: Optional[List[str]] = None
    query: Optional[str] = "cat:cs.AI"
    max_results: int = 5

class IngestResponse(BaseModel):
    task_id: str
    status: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
