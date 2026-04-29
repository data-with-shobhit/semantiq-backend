"""POST /query — full RAG pipeline + SSE streaming."""
from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from api.auth import get_tenant
from api.models.query import QueryRequest, QueryResponse
from api.metrics import (inc_cache_hit, inc_cache_miss, inc_query,
                         inc_threshold_fail, observe_query_latency)
from cache.redis_client import cache_get, cache_set
from config.logging import get_logger
from db.postgres import fetch_one, get_workspace, record_usage
from graph.query_graph import build_graph, post_langsmith_metrics
from ingestion.embedder import Embedder
from llm.anthropic_backend import get_llm_backend

log = get_logger()
router = APIRouter(prefix="/query", tags=["query"])

_llm = get_llm_backend()

_INITIAL_STATE_DEFAULTS = {
    "hyde_query": "",
    "expanded_query": "",
    "chunks": [],
    "all_chunks": [],
    "answer": "",
    "trace": [],
    "iteration": 0,
    "sub_questions": [],
    "refined_query": "",
    "llm_calls": 0,
    "tokens_used": 0,
}



@router.post("", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    tenant_id: str = Depends(get_tenant),
) -> QueryResponse:
    t0 = time.perf_counter()
    try:
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        ws = await get_workspace(req.workspace_id, tenant_id)
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")

        cache_key = f"{tenant_id}:{req.workspace_id}:{req.question}"
        if not req.bypass_cache:
            cached = cache_get(tenant_id, cache_key)
            if cached:
                inc_cache_hit(tenant_id)
                inc_query(tenant_id, "ok")
                return QueryResponse(answer=cached, chunks=[], cached=True)

        inc_cache_miss(tenant_id)
        tenant = await fetch_one("SELECT search_context, web_search_enabled FROM tenants WHERE id=$1", tenant_id)
        if not tenant:
            inc_query(tenant_id, "error")
            raise HTTPException(status_code=404, detail="Tenant not found")

        embedder = Embedder(ws["domain"])
        graph = build_graph(_llm, embedder)

        run_id = str(uuid.uuid4())
        result = await graph.ainvoke(
            {
                "tenant_id": tenant_id,
                "workspace_id": req.workspace_id,
                "collection_name": ws["collection_name"],
                "domain": ws["domain"],
                "question": req.question,
                "history": req.history,
                "search_context": tenant["search_context"] or "",
                "web_search_enabled": req.web_search if req.web_search is not None else bool(tenant["web_search_enabled"]),
                "use_hyde": req.use_hyde,
                **_INITIAL_STATE_DEFAULTS,
            },
            config={"run_id": run_id},
        )

        answer: str = result["answer"]
        chunks: list[dict] = result.get("chunks", [])
        trace: list[str] = result.get("trace", [])
        llm_calls: int = result.get("llm_calls", 0)
        tokens_used: int = result.get("tokens_used", 0)

        if not chunks:
            inc_threshold_fail(tenant_id)

        if answer and "don't have that information" not in answer:
            cache_set(tenant_id, cache_key, answer)

        used_web = any(c.get("web") for c in chunks)
        post_langsmith_metrics(tenant_id, run_id, answer, chunks, tokens_used, llm_calls, used_web)

        await record_usage(tenant_id, "query", tokens_used=tokens_used or _estimate_tokens(req.question, answer))

        latency_ms = round((time.perf_counter() - t0) * 1000)
        inc_query(tenant_id, "ok")
        log.info("query.done", tenant_id=tenant_id, chunks=len(chunks), latency_ms=latency_ms, llm_calls=llm_calls, run_id=run_id)
        return QueryResponse(answer=answer, chunks=chunks, cached=False, trace=trace, llm_calls=llm_calls, run_id=run_id)

    except HTTPException:
        raise
    except Exception as exc:
        inc_query(tenant_id, "error")
        log.error("query.failed", tenant_id=tenant_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Query pipeline failed") from exc
    finally:
        observe_query_latency(tenant_id, time.perf_counter() - t0)
        log.debug("query.exit", tenant_id=tenant_id)


@router.post("/stream")
async def query_stream(
    req: QueryRequest,
    tenant_id: str = Depends(get_tenant),
) -> StreamingResponse:
    """Fix #6: SSE streaming — node status events + answer tokens."""

    async def event_generator():
        t0 = time.perf_counter()
        try:
            if not req.question.strip():
                yield _sse({"type": "error", "data": "Question cannot be empty"})
                return

            ws = await get_workspace(req.workspace_id, tenant_id)
            if not ws:
                yield _sse({"type": "error", "data": "Workspace not found"})
                return

            stream_cache_key = f"{tenant_id}:{req.workspace_id}:{req.question}"
            if not req.bypass_cache:
                cached = cache_get(tenant_id, stream_cache_key)
                if cached:
                    yield _sse({"type": "status", "node": "cache_hit"})
                    yield _sse({"type": "answer", "data": cached, "cached": True})
                    yield _sse({"type": "done", "trace": ["cache_hit"]})
                    return

            tenant = await fetch_one("SELECT search_context, web_search_enabled FROM tenants WHERE id=$1", tenant_id)
            if not tenant:
                yield _sse({"type": "error", "data": "Tenant not found"})
                return

            embedder = Embedder(ws["domain"])
            graph = build_graph(_llm, embedder)

            run_id = str(uuid.uuid4())
            initial_state = {
                "tenant_id": tenant_id,
                "workspace_id": req.workspace_id,
                "collection_name": ws["collection_name"],
                "domain": ws["domain"],
                "question": req.question,
                "history": req.history,
                "search_context": tenant["search_context"] or "",
                "web_search_enabled": req.web_search if req.web_search is not None else bool(tenant["web_search_enabled"]),
                "use_hyde": req.use_hyde,
                **_INITIAL_STATE_DEFAULTS,
            }

            # Stream node-level events via LangGraph astream_events
            answer = ""
            trace = []
            chunks = []
            llm_calls = 0
            tokens_used = 0

            async for event in graph.astream_events(initial_state, version="v2", config={"run_id": run_id}):
                name = event.get("name", "")
                kind = event.get("event", "")

                # Node started
                if kind == "on_chain_start" and name in ("decompose", "rewrite", "retrieve", "generate", "grade", "web_search"):
                    yield _sse({"type": "status", "node": name})

                # Node completed — collect outputs (generate may fire multiple times in CRAG retry)
                if kind == "on_chain_end" and name in ("decompose", "rewrite", "retrieve", "generate", "grade", "web_search"):
                    output = event.get("data", {}).get("output", {})
                    if name == "generate" and output.get("answer"):
                        answer = output["answer"]  # always take latest, stream once after loop
                    if output.get("trace"):
                        trace = output["trace"]
                    if output.get("chunks"):
                        chunks = output["chunks"]
                    if output.get("llm_calls"):
                        llm_calls = output["llm_calls"]
                    if output.get("tokens_used"):
                        tokens_used = output["tokens_used"]

            # Stream final answer once after graph completes — prevents duplication on retries
            if answer:
                for i in range(0, len(answer), 20):
                    yield _sse({"type": "token", "data": answer[i:i + 20]})

            latency_ms = round((time.perf_counter() - t0) * 1000)
            yield _sse({"type": "done", "trace": trace, "llm_calls": llm_calls, "latency_ms": latency_ms})

            # Post-stream: LangSmith + cache + usage (best-effort)
            used_web = any(c.get("web") for c in chunks)
            post_langsmith_metrics(tenant_id, run_id, answer, chunks, tokens_used, llm_calls, used_web)
            if answer and "don't have that information" not in answer:
                cache_set(tenant_id, req.question, answer)
            final_tokens = tokens_used or _estimate_tokens(req.question, answer)
            await record_usage(tenant_id, "query", tokens_used=final_tokens)
            log.info("query.stream_done", tenant_id=tenant_id, latency_ms=latency_ms, run_id=run_id)

        except Exception as exc:
            log.error("query.stream_failed", tenant_id=tenant_id, error=str(exc))
            yield _sse({"type": "error", "data": "Query pipeline failed"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


def _estimate_tokens(question: str, answer: str) -> int:
    try:
        return (len(question) + len(answer)) // 4
    except Exception:
        return 0
