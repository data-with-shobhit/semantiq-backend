"""FastAPI application entrypoint."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from api.metrics import metrics_response
from api.routes import auth, eval, ingest, query, tenants, workspaces
from config.logging import configure_logging, get_logger
from config.settings import settings
from db.postgres import close_pool, init_pool
from ingestion.embedder import Embedder
from retrieval.splade import _load_splade

configure_logging(settings.log_level, settings.log_format)
log = get_logger()

# Export LangSmith vars to os.environ so LangChain SDK picks them up
if settings.langchain_tracing_v2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project


def _prewarm_models() -> None:
    try:
        loop = asyncio.new_event_loop()
        loop.close()
        _load_splade()
        Embedder("financial")
        Embedder("general")
        log.info("app.models_prewarmed")
    except Exception as exc:
        log.warning("app.prewarm_failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        await init_pool()
        await asyncio.get_event_loop().run_in_executor(None, _prewarm_models)
        log.info("app.startup", llm_provider=settings.llm_provider, log_format=settings.log_format)
        yield
    except Exception as exc:
        log.error("app.startup_failed", error=str(exc))
        raise
    finally:
        try:
            await close_pool()
            log.info("app.shutdown")
        except Exception as exc:
            log.error("app.shutdown_failed", error=str(exc))


app = FastAPI(title="RAG Platform", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(tenants.router)
app.include_router(workspaces.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(eval.router)


@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> Response:
    try:
        return metrics_response()
    except Exception as exc:
        log.error("metrics.failed", error=str(exc))
        raise


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    try:
        return {"status": "ok"}
    except Exception as exc:
        log.error("health.failed", error=str(exc))
        raise
