"""Prometheus metrics for the RAG platform."""
from __future__ import annotations

from fastapi import Response
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Gauge, Histogram,
                               generate_latest)

_QUERY_TOTAL = Counter(
    "rag_queries_total",
    "Total query requests",
    ["tenant_id", "status"],
)

_QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "Query end-to-end latency",
    ["tenant_id"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

_INGEST_TOTAL = Counter(
    "rag_ingest_total",
    "Total ingestion requests",
    ["tenant_id", "status"],
)

_INGEST_CHUNKS = Histogram(
    "rag_ingest_chunks",
    "Chunks produced per document",
    ["tenant_id"],
    buckets=[1, 5, 10, 25, 50, 100, 250, 500],
)

_CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Semantic cache hits",
    ["tenant_id"],
)

_CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Semantic cache misses",
    ["tenant_id"],
)

_RERANK_THRESHOLD_FAIL = Counter(
    "rag_rerank_threshold_fail_total",
    "Queries where reranker threshold gate rejected all chunks",
    ["tenant_id"],
)

_ACTIVE_TENANTS = Gauge(
    "rag_active_tenants",
    "Number of tenants with at least one document",
)


def inc_query(tenant_id: str, status: str = "ok") -> None:
    _QUERY_TOTAL.labels(tenant_id=tenant_id, status=status).inc()


def observe_query_latency(tenant_id: str, seconds: float) -> None:
    _QUERY_LATENCY.labels(tenant_id=tenant_id).observe(seconds)


def inc_ingest(tenant_id: str, status: str = "ok") -> None:
    _INGEST_TOTAL.labels(tenant_id=tenant_id, status=status).inc()


def observe_chunks(tenant_id: str, n: int) -> None:
    _INGEST_CHUNKS.labels(tenant_id=tenant_id).observe(n)


def inc_cache_hit(tenant_id: str) -> None:
    _CACHE_HITS.labels(tenant_id=tenant_id).inc()


def inc_cache_miss(tenant_id: str) -> None:
    _CACHE_MISSES.labels(tenant_id=tenant_id).inc()


def inc_threshold_fail(tenant_id: str) -> None:
    _RERANK_THRESHOLD_FAIL.labels(tenant_id=tenant_id).inc()


def set_active_tenants(n: int) -> None:
    _ACTIVE_TENANTS.set(n)


def metrics_response() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
