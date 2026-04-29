"""asyncpg connection pool + typed query helpers."""
from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import asyncpg

from config.logging import get_logger
from config.settings import settings

log = get_logger()

_pool: asyncpg.Pool | None = None


async def init_pool() -> None:
    global _pool
    t0 = time.perf_counter()
    try:
        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=settings.postgres_min_pool,
            max_size=settings.postgres_max_pool,
            statement_cache_size=0,  # required for Supabase PgBouncer transaction mode
        )
        log.info("postgres.pool_created", elapsed_ms=round((time.perf_counter() - t0) * 1000))
    except Exception as exc:
        log.error("postgres.pool_failed", error=str(exc))
        raise
    finally:
        pass  # pool is long-lived; no per-call teardown


async def close_pool() -> None:
    global _pool
    try:
        if _pool:
            await _pool.close()
            log.info("postgres.pool_closed")
    except Exception as exc:
        log.error("postgres.pool_close_failed", error=str(exc))
    finally:
        _pool = None


def _get_pool() -> asyncpg.Pool:
    try:
        if _pool is None:
            raise RuntimeError("Postgres pool not initialised — call init_pool() first")
        return _pool
    except RuntimeError:
        raise


async def fetch_one(query: str, *args: Any) -> asyncpg.Record | None:
    try:
        return await _get_pool().fetchrow(query, *args)
    except asyncpg.PostgresError as exc:
        log.error("postgres.fetch_one_failed", query=query[:80], error=str(exc))
        raise
    except Exception as exc:
        log.error("postgres.fetch_one_error", query=query[:80], error=str(exc))
        raise


async def fetch_all(query: str, *args: Any) -> list[asyncpg.Record]:
    try:
        return await _get_pool().fetch(query, *args)
    except asyncpg.PostgresError as exc:
        log.error("postgres.fetch_all_failed", query=query[:80], error=str(exc))
        raise
    except Exception as exc:
        log.error("postgres.fetch_all_error", query=query[:80], error=str(exc))
        raise


async def execute(query: str, *args: Any) -> str:
    try:
        return await _get_pool().execute(query, *args)
    except asyncpg.PostgresError as exc:
        log.error("postgres.execute_failed", query=query[:80], error=str(exc))
        raise
    except Exception as exc:
        log.error("postgres.execute_error", query=query[:80], error=str(exc))
        raise


async def execute_returning(query: str, *args: Any) -> asyncpg.Record:
    try:
        row = await _get_pool().fetchrow(query, *args)
        if row is None:
            raise RuntimeError("execute_returning returned no row")
        return row
    except asyncpg.PostgresError as exc:
        log.error("postgres.execute_returning_failed", query=query[:80], error=str(exc))
        raise
    except Exception as exc:
        log.error("postgres.execute_returning_error", query=query[:80], error=str(exc))
        raise


@asynccontextmanager
async def transaction() -> AsyncIterator[asyncpg.Connection]:
    conn = None
    try:
        conn = await _get_pool().acquire()
        async with conn.transaction():
            yield conn
    except Exception as exc:
        log.error("postgres.transaction_failed", error=str(exc))
        raise
    finally:
        if conn is not None:
            await _get_pool().release(conn)


_STORAGE_QUOTA_BYTES = 200 * 1024 * 1024  # 200 MB per tenant


# ── Workspace helpers ─────────────────────────────────────────────────────────

async def create_workspace(tenant_id: str, name: str, domain: str, embed_model: str, embed_dim: int) -> asyncpg.Record:
    try:
        row = await execute_returning(
            """
            INSERT INTO workspaces (tenant_id, name, domain, embed_model, embed_dim, collection_name)
            VALUES ($1, $2, $3, $4, $5, $1 || '_' || currval('workspaces_id_seq') || '_docs')
            RETURNING *
            """,
            tenant_id, name, domain, embed_model, embed_dim,
        )
        # collection_name uses the generated id — update after insert
        ws_id = row["id"]
        collection = f"{tenant_id}_{ws_id}_docs"
        await execute(
            "UPDATE workspaces SET collection_name=$1 WHERE id=$2",
            collection, ws_id,
        )
        return await fetch_one("SELECT * FROM workspaces WHERE id=$1", ws_id)
    except asyncpg.UniqueViolationError:
        raise
    except Exception as exc:
        log.error("postgres.create_workspace_failed", tenant_id=tenant_id, name=name, error=str(exc))
        raise


async def get_workspace(workspace_id: int, tenant_id: str) -> asyncpg.Record | None:
    try:
        return await fetch_one(
            "SELECT * FROM workspaces WHERE id=$1 AND tenant_id=$2",
            workspace_id, tenant_id,
        )
    except Exception as exc:
        log.error("postgres.get_workspace_failed", workspace_id=workspace_id, error=str(exc))
        raise


async def list_workspaces(tenant_id: str) -> list[asyncpg.Record]:
    try:
        return await fetch_all(
            "SELECT * FROM workspaces WHERE tenant_id=$1 ORDER BY created_at DESC",
            tenant_id,
        )
    except Exception as exc:
        log.error("postgres.list_workspaces_failed", tenant_id=tenant_id, error=str(exc))
        raise


async def tenant_storage_bytes(tenant_id: str) -> int:
    try:
        row = await fetch_one(
            "SELECT COALESCE(SUM(storage_bytes), 0) AS total FROM workspaces WHERE tenant_id=$1",
            tenant_id,
        )
        return int(row["total"]) if row else 0
    except Exception as exc:
        log.error("postgres.tenant_storage_failed", tenant_id=tenant_id, error=str(exc))
        raise


async def increment_workspace_storage(workspace_id: int, bytes_delta: int) -> None:
    try:
        await execute(
            "UPDATE workspaces SET storage_bytes = storage_bytes + $1 WHERE id=$2",
            bytes_delta, workspace_id,
        )
    except Exception as exc:
        log.warning("postgres.increment_storage_failed", workspace_id=workspace_id, error=str(exc))


# ── Tenant helpers ────────────────────────────────────────────────────────────

async def get_tenant(tenant_id: str) -> asyncpg.Record | None:
    try:
        return await fetch_one("SELECT * FROM tenants WHERE id = $1", tenant_id)
    except Exception as exc:
        log.error("postgres.get_tenant_failed", tenant_id=tenant_id, error=str(exc))
        raise


async def tenant_doc_count(tenant_id: str) -> int:
    try:
        row = await fetch_one(
            "SELECT COUNT(*) AS n FROM documents WHERE tenant_id = $1", tenant_id
        )
        return int(row["n"]) if row else 0
    except Exception as exc:
        log.error("postgres.tenant_doc_count_failed", tenant_id=tenant_id, error=str(exc))
        raise


# ── Document helpers ──────────────────────────────────────────────────────────

async def create_document(tenant_id: str, workspace_id: int, filename: str, path: str, file_size_bytes: int = 0) -> int:
    try:
        row = await execute_returning(
            """
            INSERT INTO documents (tenant_id, workspace_id, filename, path, status, file_size_bytes)
            VALUES ($1, $2, $3, $4, 'processing', $5)
            RETURNING id
            """,
            tenant_id, workspace_id, filename, path, file_size_bytes,
        )
        doc_id: int = row["id"]
        log.info("document.created", tenant_id=tenant_id, workspace_id=workspace_id, doc_id=doc_id, filename=filename)
        return doc_id
    except Exception as exc:
        log.error("postgres.create_document_failed", tenant_id=tenant_id, filename=filename, error=str(exc))
        raise


async def mark_document_ready(doc_id: int, chunk_count: int, tenant_id: str) -> None:
    try:
        await execute(
            "UPDATE documents SET status='ready', chunk_count=$1 WHERE id=$2",
            chunk_count, doc_id,
        )
        log.info("document.ready", tenant_id=tenant_id, doc_id=doc_id, chunks=chunk_count)
    except Exception as exc:
        log.error("postgres.mark_ready_failed", tenant_id=tenant_id, doc_id=doc_id, error=str(exc))
        raise


async def mark_document_failed(doc_id: int, error_msg: str, tenant_id: str) -> None:
    try:
        await execute(
            "UPDATE documents SET status='failed', error_msg=$1 WHERE id=$2",
            error_msg, doc_id,
        )
        log.warning("document.failed", tenant_id=tenant_id, doc_id=doc_id, error=error_msg)
    except Exception as exc:
        log.error("postgres.mark_failed_failed", tenant_id=tenant_id, doc_id=doc_id, error=str(exc))
        raise


# ── Strategy helpers ─────────────────────────────────────────────────────────

async def save_strategy(
    doc_id: int,
    tenant_id: str,
    chunker: str,
    chunk_size: int,
    overlap: int,
    reasoning: str,
    source: str,
    extra_params: dict | None = None,
) -> int:
    try:
        await execute(
            "UPDATE ingestion_strategies SET is_active=FALSE WHERE doc_id=$1 AND tenant_id=$2",
            doc_id, tenant_id,
        )
        row = await execute_returning(
            """
            INSERT INTO ingestion_strategies
                (doc_id, tenant_id, chunker, chunk_size, overlap, extra_params, reasoning, source,
                 version)
            SELECT $1, $2, $3, $4, $5, $6, $7, $8,
                   COALESCE((SELECT MAX(version) FROM ingestion_strategies
                             WHERE doc_id=$1 AND tenant_id=$2), 0) + 1
            RETURNING id
            """,
            doc_id, tenant_id, chunker, chunk_size, overlap,
            json.dumps(extra_params or {}), reasoning, source,
        )
        return int(row["id"])
    except Exception as exc:
        log.error("postgres.save_strategy_failed", doc_id=doc_id, error=str(exc))
        raise


async def get_active_strategy(doc_id: int, tenant_id: str) -> asyncpg.Record | None:
    try:
        return await fetch_one(
            "SELECT * FROM ingestion_strategies WHERE doc_id=$1 AND tenant_id=$2 AND is_active=TRUE",
            doc_id, tenant_id,
        )
    except Exception as exc:
        log.error("postgres.get_active_strategy_failed", doc_id=doc_id, error=str(exc))
        raise


async def save_ragas_eval(
    doc_id: int,
    tenant_id: str,
    strategy_id: int | None,
    questions: list,
    ground_truths: list,
    answers: list,
    contexts: list,
    faithfulness: float,
    context_precision: float,
    answer_relevance: float,
    retrieval_metrics: dict | None = None,
) -> int:
    try:
        avg = round((faithfulness + context_precision + answer_relevance) / 3, 4)
        # Replace previous eval — one canonical result per doc at any time
        await execute(
            "DELETE FROM ragas_evals WHERE doc_id=$1 AND tenant_id=$2",
            doc_id, tenant_id,
        )
        row = await execute_returning(
            """
            INSERT INTO ragas_evals
                (doc_id, tenant_id, strategy_id, questions, ground_truths,
                 answers, contexts, faithfulness, context_precision, answer_relevance,
                 avg_score, retrieval_metrics)
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
            RETURNING id
            """,
            doc_id, tenant_id, strategy_id,
            json.dumps(questions), json.dumps(ground_truths),
            json.dumps(answers), json.dumps(contexts),
            faithfulness, context_precision, answer_relevance, avg,
            json.dumps(retrieval_metrics) if retrieval_metrics else None,
        )
        scores_json = json.dumps({
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "answer_relevance": answer_relevance,
            "avg": avg,
        })
        if strategy_id:
            await execute(
                "UPDATE ingestion_strategies SET ragas_scores=$1 WHERE id=$2",
                scores_json, strategy_id,
            )
        log.info("postgres.ragas_eval_saved", doc_id=doc_id, tenant_id=tenant_id,
                 eval_id=int(row["id"]), faithfulness=faithfulness, avg=avg)
        return int(row["id"])
    except Exception as exc:
        log.error("postgres.save_ragas_eval_failed", doc_id=doc_id, error=str(exc))
        raise


async def get_latest_ragas_eval(doc_id: int, tenant_id: str) -> asyncpg.Record | None:
    try:
        return await fetch_one(
            "SELECT * FROM ragas_evals WHERE doc_id=$1 AND tenant_id=$2 ORDER BY created_at DESC LIMIT 1",
            doc_id, tenant_id,
        )
    except Exception as exc:
        log.error("postgres.get_latest_ragas_eval_failed", doc_id=doc_id, error=str(exc))
        raise


async def revert_strategy(doc_id: int, tenant_id: str, version: int) -> bool:
    try:
        await execute(
            "UPDATE ingestion_strategies SET is_active=FALSE WHERE doc_id=$1 AND tenant_id=$2",
            doc_id, tenant_id,
        )
        result = await execute(
            "UPDATE ingestion_strategies SET is_active=TRUE WHERE doc_id=$1 AND tenant_id=$2 AND version=$3",
            doc_id, tenant_id, version,
        )
        return result != "UPDATE 0"
    except Exception as exc:
        log.error("postgres.revert_strategy_failed", doc_id=doc_id, version=version, error=str(exc))
        raise


# ── Usage helpers ─────────────────────────────────────────────────────────────

async def record_usage(tenant_id: str, event_type: str, tokens_used: int | None = None) -> None:
    try:
        await execute(
            "INSERT INTO usage (tenant_id, event_type, tokens_used) VALUES ($1, $2, $3)",
            tenant_id, event_type, tokens_used,
        )
        log.info("usage.recorded", tenant_id=tenant_id, event_type=event_type, tokens=tokens_used)
    except Exception as exc:
        log.warning("postgres.record_usage_failed", tenant_id=tenant_id, error=str(exc))
