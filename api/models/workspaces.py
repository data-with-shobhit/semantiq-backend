from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from ingestion.model_registry import MODEL_REGISTRY


class WorkspaceCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    domain: str = Field(..., description="legal | financial | medical | clinical | scientific | technical | general")

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        if v not in MODEL_REGISTRY:
            raise ValueError(f"Unknown domain '{v}'. Valid: {sorted(MODEL_REGISTRY)}")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name cannot be blank")
        return v


class WorkspaceResponse(BaseModel):
    id: int
    name: str
    domain: str
    embed_model: str
    embed_dim: int
    collection_name: str
    storage_mb: float
    created_at: str | None
