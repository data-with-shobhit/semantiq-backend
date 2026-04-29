from __future__ import annotations

from pydantic import BaseModel, field_validator

from ingestion.model_registry import DOMAINS

_PLANS = {"free", "pro", "enterprise"}


class SignupRequest(BaseModel):
    tenant_id: str
    plan: str = "free"
    domain: str = "general"
    search_context: str = ""
    web_search_enabled: bool = False

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        v = v.strip().lower()
        if not v or not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("tenant_id must be alphanumeric (dashes/underscores ok)")
        if len(v) > 64:
            raise ValueError("tenant_id must be 64 characters or fewer")
        return v

    @field_validator("plan")
    @classmethod
    def validate_plan(cls, v: str) -> str:
        if v not in _PLANS:
            raise ValueError(f"plan must be one of {sorted(_PLANS)}")
        return v

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: str) -> str:
        if v not in DOMAINS:
            raise ValueError(f"domain must be one of {sorted(DOMAINS)}")
        return v

    @field_validator("search_context")
    @classmethod
    def validate_search_context(cls, v: str) -> str:
        if len(v) > 500:
            raise ValueError("search_context must be 500 characters or fewer")
        return v


class SignupResponse(BaseModel):
    tenant_id: str
    token: str
    embed_model: str
    embed_dim: int
