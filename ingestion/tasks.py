"""Celery ingestion tasks — domain-specific worker queues."""
from __future__ import annotations

import asyncio
import os
import ssl as _ssl
import sys
import tempfile
from pathlib import Path

from celery import Celery
from celery.signals import task_failure
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from cache.redis_client import set_doc_status
from config.logging import configure_logging, get_logger
from config.settings import settings
from db.postgres import _pool, execute, fetch_one, init_pool
from ingestion.analyzer import analyze as analyze_doc
from ingestion.chunker import chunk_text
from ingestion.embedder import Embedder
from ingestion.extractor import extract_metadata
from ingestion.loader import load as _load_local
from retrieval import splade as splade_enc

sys.path.insert(0, str(Path(__file__).parent.parent))



configure_logging(settings.log_level, settings.log_format)
log = get_logger()

_SSL_OPTS = {"ssl_cert_reqs": _ssl.CERT_NONE} if settings.celery_broker_url.startswith("rediss://") else {}

celery_app = Celery(
    "ingestion",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
if _SSL_OPTS:
    celery_app.conf.broker_use_ssl = _SSL_OPTS
    celery_app.conf.redis_backend_use_ssl = _SSL_OPTS

# Upstash drops idle connections — heartbeat + retry keeps worker alive on Windows
celery_app.conf.broker_transport_options = {
    "visibility_timeout": 3600,
    "retry_on_timeout": True,
    "max_connections": 5,
}
celery_app.conf.broker_connection_retry = True
celery_app.conf.broker_connection_retry_on_startup = True
celery_app.conf.broker_connection_max_retries = None
celery_app.conf.broker_heartbeat = 10        # send heartbeat every 10s
celery_app.conf.broker_heartbeat_checkrate = 2
celery_app.conf.worker_proc_alive_timeout = 60

# Dead-letter queue — tasks that exhaust all retries land here instead of vanishing
celery_app.conf.task_routes = {
    "ingestion.tasks.process_document": {"queue": "ingestion"},
}
celery_app.conf.task_queues_max_priority = 10
# Failed tasks sent to dead-letter queue after max_retries exhausted
celery_app.conf.task_reject_on_worker_lost = True
celery_app.conf.task_acks_late = True


def _qdrant() -> QdrantClient:
    try:
        kwargs: dict = {
            "host": settings.qdrant_host,
            "port": settings.qdrant_port,
            "timeout": 120,
        }
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
            kwargs["https"] = True
        return QdrantClient(**kwargs)
    except Exception as exc:
        log.error("tasks.qdrant_connect_failed", error=str(exc))
        raise


def _pg_run(coro):
    """Run an asyncpg coroutine from sync Celery context."""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            if _pool is None:
                loop.run_until_complete(init_pool())
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    except Exception as exc:
        log.error("tasks.pg_run_failed", error=str(exc))
        raise


def _get_tenant_domain(tenant_id: str) -> str:
    try:
        row = _pg_run(fetch_one("SELECT domain FROM tenants WHERE id=$1", tenant_id))
        if not row:
            raise ValueError(f"Tenant not found: {tenant_id}")
        return row["domain"]
    except Exception as exc:
        log.error("tasks.get_tenant_domain_failed", tenant_id=tenant_id, error=str(exc))
        raise
    finally:
        log.debug("tasks.get_tenant_domain_exit", tenant_id=tenant_id)


def _update_doc_status(
    doc_id: int,
    status: str,
    chunk_count: int | None = None,
    error: str | None = None,
) -> None:
    try:
        if chunk_count is not None:
            _pg_run(execute(
                "UPDATE documents SET status=$1, chunk_count=$2 WHERE id=$3",
                status, chunk_count, doc_id,
            ))
        elif error is not None:
            _pg_run(execute(
                "UPDATE documents SET status=$1, error_msg=$2 WHERE id=$3",
                status, error, doc_id,
            ))
        else:
            _pg_run(execute("UPDATE documents SET status=$1 WHERE id=$2", status, doc_id))
    except Exception as exc:
        log.warning("tasks.update_doc_status_failed", doc_id=doc_id, status=status, error=str(exc))
    finally:
        log.debug("tasks.update_doc_status_exit", doc_id=doc_id, status=status)


def _chunk_id(doc_id: int, chunk_index: int) -> int:
    try:
        return doc_id * 100_000 + chunk_index
    except Exception as exc:
        log.error("tasks.chunk_id_failed", doc_id=doc_id, chunk_index=chunk_index, error=str(exc))
        raise


@task_failure.connect
def _on_permanent_failure(task_id: str, exception: Exception, args: tuple, kwargs: dict, traceback, einfo, **kw) -> None:
    """Fires after all retries exhausted — the dead-letter handler."""
    sender_name = kw.get("sender", {})
    if hasattr(sender_name, "name"):
        sender_name = sender_name.name
    if "process_document" not in str(sender_name):
        return
    tenant_id = args[0] if args else "unknown"
    doc_id = args[1] if len(args) > 1 else -1
    log.error(
        "task.dead_letter",
        task_id=task_id,
        tenant_id=tenant_id,
        doc_id=doc_id,
        error=str(exception),
        traceback=str(einfo),
    )
    try:
        _update_doc_status(doc_id, "failed", error=f"Permanently failed after all retries: {exception}")
        set_doc_status(tenant_id, doc_id, "failed")
    except Exception as mark_exc:
        log.warning("task.dead_letter_mark_failed", doc_id=doc_id, error=str(mark_exc))


@celery_app.task(bind=True, max_retries=3)
def process_document(self, tenant_id: str, doc_id: int, file_path: str, skip_analyze: bool = False, workspace_id: int | None = None) -> dict:
    log.info("task.started", tenant_id=tenant_id, doc_id=doc_id)
    set_doc_status(tenant_id, doc_id, "processing")

    try:
        # Resolve domain from workspace if provided, else fall back to tenant domain
        if workspace_id is not None:
            ws_row = _pg_run(fetch_one("SELECT domain, collection_name FROM workspaces WHERE id=$1 AND tenant_id=$2", workspace_id, tenant_id))
            domain = ws_row["domain"] if ws_row else _get_tenant_domain(tenant_id)
            collection_name = ws_row["collection_name"] if ws_row else f"{tenant_id}_docs"
        else:
            domain = _get_tenant_domain(tenant_id)
            collection_name = f"{tenant_id}_docs"
        if file_path.startswith("r2://"):
            from storage.r2 import R2Storage
            data = R2Storage().load(file_path)
            suffix = os.path.splitext(file_path)[1] or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            text = _load_local(tmp_path)
            os.unlink(tmp_path)
        else:
            text = _load_local(file_path)

        # Analyze structure — skip if strategy was cloned from another doc
        filename = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        from db.postgres import get_active_strategy, save_strategy
        if skip_analyze:
            existing = _pg_run(get_active_strategy(doc_id, tenant_id))
            if existing:
                from ingestion.analyzer import IngestionStrategy
                strategy = IngestionStrategy(
                    chunker=existing["chunker"],
                    chunk_size=existing["chunk_size"],
                    overlap=existing["overlap"],
                    reasoning=existing["reasoning"] or "Cloned strategy",
                    confidence=1.0,
                    source="user_revert",
                )
                log.info("task.strategy_reused", tenant_id=tenant_id, doc_id=doc_id,
                         chunker=strategy.chunker)
            else:
                skip_analyze = False

        if not skip_analyze:
            strategy = analyze_doc(text, domain, filename)
            log.info(
                "task.strategy",
                tenant_id=tenant_id, doc_id=doc_id,
                chunker=strategy.chunker, chunk_size=strategy.chunk_size,
                overlap=strategy.overlap, confidence=strategy.confidence,
                source=strategy.source,
            )
            _pg_run(
                save_strategy(
                    doc_id, tenant_id,
                    strategy.chunker, strategy.chunk_size, strategy.overlap,
                    strategy.reasoning, strategy.source, strategy.extra_params,
                )
            )

        chunks = chunk_text(text, domain=domain, strategy=strategy)
        base_meta = extract_metadata(text, file_path, domain=domain)

        if not chunks:
            log.warning("task.no_chunks", tenant_id=tenant_id, doc_id=doc_id, file_path=file_path)
            _update_doc_status(doc_id, "failed", error="PDF extracted 0 chunks — likely image-based PDF with no selectable text")
            set_doc_status(tenant_id, doc_id, "failed")
            return {"doc_id": doc_id, "chunks": 0}

        embedder = Embedder(domain)
        texts = [c["text"] for c in chunks]
        dense_vectors = embedder.encode(texts)
        sparse_vectors = splade_enc.encode_batch(texts)

        points = []
        for i, (chunk, dense, (sp_idx, sp_val)) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        ):
            meta = {
                **base_meta,
                **chunk["metadata"],
                "doc_id": doc_id,
                "tenant_id": tenant_id,
                "text": chunk["text"],
            }
            points.append(PointStruct(
                id=_chunk_id(doc_id, i),
                vector={
                    "dense": dense,
                    "sparse": SparseVector(indices=sp_idx, values=sp_val),
                },
                payload=meta,
            ))

        # Fill prev/next chunk IDs now that all point IDs are known
        for i, point in enumerate(points):
            point.payload["prev_chunk_id"] = points[i - 1].id if i > 0 else None
            point.payload["next_chunk_id"] = points[i + 1].id if i < len(points) - 1 else None

        client = _qdrant()
        batch_size = 16
        for i in range(0, len(points), batch_size):
            client.upsert(collection_name=collection_name, points=points[i:i + batch_size])
            log.debug("task.upsert_batch", tenant_id=tenant_id, batch=i // batch_size, total=len(points))
        _update_doc_status(doc_id, "ready", chunk_count=len(chunks))
        set_doc_status(tenant_id, doc_id, "ready")
        log.info("task.done", tenant_id=tenant_id, doc_id=doc_id, chunks=len(chunks))
        return {"doc_id": doc_id, "chunks": len(chunks)}

    except Exception as exc:
        log.error("task.failed", tenant_id=tenant_id, doc_id=doc_id, error=str(exc))
        _update_doc_status(doc_id, "failed", error=str(exc))
        set_doc_status(tenant_id, doc_id, "failed")
        raise self.retry(exc=exc, countdown=30)

    finally:
        log.debug("task.exit", tenant_id=tenant_id, doc_id=doc_id)
