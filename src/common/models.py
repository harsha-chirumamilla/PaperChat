from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_cache: bool = True
    chat_history: Optional[List[Dict[str, str]]] = None

class QueryResponse(BaseModel):
    task_id: str
    status: str

class IngestRequest(BaseModel):
    # For now, simple ingest from Arxiv query or similar
    # Or maybe list of arxiv IDs?
    # Improvement plan mentions "POST /api/v1/ingest/arxiv - Batch download"
    arxiv_ids: Optional[List[str]] = None
    query: Optional[str] = "cat:cs.AI"
    max_results: int = Field(default=5, ge=1, le=1000000)
    skip_papers: int = Field(default=0, ge=0, le=1000000)

class IngestResponse(BaseModel):
    task_id: str
    status: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None

class SuggestQuestionsRequest(BaseModel):
    current_query: str
    answer: str
    chat_history: Optional[List[Dict[str, str]]] = None

class SuggestQuestionsResponse(BaseModel):
    task_id: str
    status: str
