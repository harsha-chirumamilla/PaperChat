from pydantic import BaseModel, Field
from typing import Optional, Any, Dict

class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_cache: bool = True
    filter_metadata: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    task_id: str
    status: str

class IngestRequest(BaseModel):
    query: str = Field(default="cat:cs.AI", min_length=1, max_length=500)
    max_results: int = Field(default=5, ge=1, le=100)
    skip_papers: int = Field(default=0, ge=0, le=100000)

class IngestResponse(BaseModel):
    task_id: str
    status: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
