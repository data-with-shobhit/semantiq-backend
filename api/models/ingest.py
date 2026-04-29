from __future__ import annotations

from pydantic import BaseModel


class IngestResponse(BaseModel):
    job_id: str
    doc_id: int
    filename: str
    strategy_reused: str | None = None
    status: str = "processing"
