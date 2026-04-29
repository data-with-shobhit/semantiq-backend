from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    workspace_id: int = Field(..., gt=0)
    history: list[dict] = Field(default_factory=list, max_length=20)
    filters: dict | None = None
    bypass_cache: bool = False
    use_hyde: bool = True
    web_search: bool | None = None


class QueryResponse(BaseModel):
    answer: str
    chunks: list[dict]
    cached: bool = False
    trace: list[str] = []
    llm_calls: int = 0
    run_id: str | None = None
