"""Centralised configuration — single source of truth for all env vars.

Usage anywhere in the codebase:
    from config.settings import settings
    url = settings.redis_url
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── JWT ──────────────────────────────────────────────────────────────────
    jwt_secret: str = Field(..., min_length=16, description="HS256 signing secret")
    jwt_algorithm: str = "HS256"
    jwt_expiry_days: int = Field(30, ge=1, le=365)

    # ── Postgres ─────────────────────────────────────────────────────────────
    database_url: str = Field(..., description="asyncpg DSN: postgresql://user:pass@host/db")
    postgres_min_pool: int = Field(2, ge=1)
    postgres_max_pool: int = Field(10, ge=1)

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = Field("redis://localhost:6379/0")
    celery_broker_url: str = Field("redis://localhost:6379/1")
    celery_result_backend: str = Field("redis://localhost:6379/2")

    # ── Qdrant ────────────────────────────────────────────────────────────────
    qdrant_host: str = Field("localhost")
    qdrant_port: int = Field(6333, ge=1, le=65535)
    qdrant_grpc_port: int = Field(6334, ge=1, le=65535)
    qdrant_api_key: str = ""

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["anthropic", "groq", "openrouter", "gemini"] = "groq"
    anthropic_api_key: str = ""
    groq_api_key: str = ""
    groq_api_key_2: str = ""
    groq_generation_model: str = "llama-3.3-70b-versatile"
    groq_eval_model: str = "openai/gpt-oss-120b"
    openrouter_api_key: str = ""
    openrouter_model: str = "google/gemma-3-27b-it:free"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # ── Embeddings ────────────────────────────────────────────────────────────
    jina_api_key: str = ""
    jina_model: str = "jina-embeddings-v3"
    voyage_api_key: str = ""
    voyage_reranker_model: str = "rerank-2.5"
    voyage_embedding_model: str = "voyage-4-lite"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    score_threshold: float = Field(0.72, ge=0.0, le=1.0)
    max_chunks: int = Field(5, ge=1, le=20)
    rrf_k: int = Field(60, ge=1)
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_hyde: bool = False

    # ── Storage ───────────────────────────────────────────────────────────────
    storage_backend: Literal["local", "r2"] = "local"
    storage_root: str = "storage/docs"
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket_name: str = "rag-docs"

    # ── Google OAuth ─────────────────────────────────────────────────────────
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/auth/google/callback"
    frontend_url: str = "http://localhost:3000"

    # ── API ───────────────────────────────────────────────────────────────────
    api_base_url: str = "http://localhost:8000"

    # ── Web Search ───────────────────────────────────────────────────────────
    tavily_api_key: str = ""

    # ── Observability ─────────────────────────────────────────────────────────
    log_format: Literal["json", "console"] = "console"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "rag-platform"

    @field_validator("jwt_secret")
    @classmethod
    def warn_weak_secret(cls, v: str) -> str:
        try:
            if v in ("change-me-before-production", "dev-secret-change-me", "secret"):
                import warnings
                warnings.warn(
                    "JWT_SECRET is a known weak default — change before deploying to production.",
                    stacklevel=2,
                )
            return v
        except Exception as exc:
            raise ValueError(f"jwt_secret validation failed: {exc}") from exc

    @field_validator("database_url")
    @classmethod
    def validate_dsn(cls, v: str) -> str:
        try:
            if not v.startswith(("postgresql://", "postgres://")):
                raise ValueError("DATABASE_URL must start with postgresql:// or postgres://")
            return v
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"database_url validation failed: {exc}") from exc

    @model_validator(mode="after")
    def validate_anthropic_key(self) -> "Settings":
        try:
            if self.llm_provider == "anthropic" and not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
            return self
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"settings cross-validation failed: {exc}") from exc

    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
