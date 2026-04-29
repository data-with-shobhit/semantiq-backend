"""POST /ingest — upload file, store, enqueue Celery task. DELETE /ingest/{doc_id} — remove doc + chunks. GET /ingest/list — list tenant docs."""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

from api.auth import get_tenant
from api.metrics import inc_ingest
from api.models.ingest import IngestResponse
from config.logging import get_logger
from config.settings import settings
from config.settings import settings as _settings
from db.postgres import (_STORAGE_QUOTA_BYTES, create_document, execute,
                         fetch_all, fetch_one, get_workspace,
                         increment_workspace_storage, save_strategy,
                         tenant_doc_count, tenant_storage_bytes)
from ingestion.tasks import process_document
from storage.local import LocalStorage
from storage.r2 import R2Storage

log = get_logger()
router = APIRouter(prefix="/ingest", tags=["ingest"])

_storage = R2Storage() if _settings.storage_backend == "r2" else LocalStorage()
_ALLOWED = {".pdf", ".txt", ".md", ".docx", ".py", ".js", ".ts"}



@router.post("", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    file: UploadFile,
    workspace_id: int,
    tenant_id: str = Depends(get_tenant),
    reuse_strategy_id: int | None = None,
    reuse_strategy_name: str | None = None,
) -> IngestResponse:
    try:
        _validate_file(file)

        # Verify workspace belongs to tenant
        ws = await get_workspace(workspace_id, tenant_id)
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")

        # 200 MB hard quota — block and tell user to free space
        file_size = file.size or 0
        used = await tenant_storage_bytes(tenant_id)
        if used + file_size > _STORAGE_QUOTA_BYTES:
            used_mb = round(used / (1024 * 1024), 1)
            free_mb = round((_STORAGE_QUOTA_BYTES - used) / (1024 * 1024), 1)
            raise HTTPException(
                status_code=413,
                detail={
                    "error": "storage_quota_exceeded",
                    "message": f"Storage full ({used_mb} MB used, {free_mb} MB remaining). Delete a workspace or document to free space.",
                    "used_mb": used_mb,
                    "quota_mb": 200,
                    "free_mb": free_mb,
                },
            )

        await _check_quota(tenant_id)
        data = await file.read()
        path = _storage.save(tenant_id, file.filename, data)
        doc_id = await create_document(tenant_id, workspace_id, file.filename, path, file_size)

        # Track storage on workspace
        await increment_workspace_storage(workspace_id, file_size)

        # Resolve reuse strategy if requested
        cloned_name: str | None = None
        if reuse_strategy_id or reuse_strategy_name:
            cloned_name = await _clone_strategy(tenant_id, doc_id, reuse_strategy_id, reuse_strategy_name)

        task = process_document.apply_async(
            args=[tenant_id, doc_id, path],
            kwargs={"skip_analyze": cloned_name is not None, "workspace_id": workspace_id},
            queue=await _domain_queue(tenant_id, ws["domain"]),
            task_id=f"{tenant_id}-doc-{doc_id}",
        )

        inc_ingest(tenant_id, "ok")
        log.info("ingest.enqueued", tenant_id=tenant_id, workspace_id=workspace_id,
                 doc_id=doc_id, job_id=task.id, strategy_reused=cloned_name)
        return IngestResponse(job_id=task.id, doc_id=doc_id, filename=file.filename,
                              strategy_reused=cloned_name)

    except HTTPException:
        raise
    except Exception as exc:
        inc_ingest(tenant_id, "error")
        log.error("ingest.failed", tenant_id=tenant_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Ingestion failed") from exc
    finally:
        log.debug("ingest.exit", tenant_id=tenant_id)


async def _clone_strategy(
    tenant_id: str, new_doc_id: int,
    strategy_id: int | None, strategy_name: str | None,
) -> str | None:
    """Copy an existing strategy row to a new doc. Returns strategy name/id used."""
    if strategy_name:
        row = await fetch_one(
            "SELECT * FROM ingestion_strategies WHERE tenant_id=$1 AND name=$2 ORDER BY created_at DESC LIMIT 1",
            tenant_id, strategy_name,
        )
    else:
        row = await fetch_one(
            "SELECT * FROM ingestion_strategies WHERE id=$1 AND tenant_id=$2",
            strategy_id, tenant_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="Referenced strategy not found")

    extra = json.loads(row["extra_params"]) if row["extra_params"] else {}
    sid = await save_strategy(
        new_doc_id, tenant_id,
        row["chunker"], row["chunk_size"], row["overlap"],
        f"Cloned from strategy '{row['name'] or row['id']}'", "user_revert", extra,
    )
    # Preserve name if the source had one
    if row["name"]:
        await execute(
            "UPDATE ingestion_strategies SET name=$1 WHERE id=$2",
            f"{row['name']} (copy)", sid,
        )
    label = row["name"] or str(row["id"])
    log.info("ingest.strategy_cloned", tenant_id=tenant_id, doc_id=new_doc_id,
             from_strategy=label, new_strategy_id=sid)
    return label


@router.get("/list", status_code=status.HTTP_200_OK)
async def list_documents(
    workspace_id: int,
    tenant_id: str = Depends(get_tenant),
) -> dict[str, Any]:
    try:
        ws = await get_workspace(workspace_id, tenant_id)
        if not ws:
            raise HTTPException(status_code=404, detail="Workspace not found")
        rows = await fetch_all(
            "SELECT id, filename, status, chunk_count, uploaded_at FROM documents WHERE tenant_id=$1 AND workspace_id=$2 ORDER BY uploaded_at DESC",
            tenant_id, workspace_id,
        )
        docs = [
            {
                "id": r["id"],
                "filename": r["filename"],
                "status": r["status"],
                "chunk_count": r["chunk_count"],
                "uploaded_at": r["uploaded_at"].isoformat() if r["uploaded_at"] else None,
            }
            for r in rows
        ]
        return {"documents": docs}
    except HTTPException:
        raise
    except Exception as exc:
        log.error("ingest.list_failed", tenant_id=tenant_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to list documents") from exc


@router.delete("/{doc_id}", status_code=status.HTTP_200_OK)
async def delete_document(
    doc_id: int,
    tenant_id: str = Depends(get_tenant),
) -> dict:
    try:
        doc = await fetch_one(
            "SELECT id, filename, path, chunk_count, status FROM documents WHERE id=$1 AND tenant_id=$2",
            doc_id, tenant_id,
        )
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        chunk_count = doc["chunk_count"] or 0
        deleted_chunks = 0

        # Delete Qdrant points — only this doc's chunks, not the whole collection
        if chunk_count > 0:
            point_ids = list(range(doc_id * 100_000, doc_id * 100_000 + chunk_count))
            kwargs: dict = {"host": settings.qdrant_host, "port": settings.qdrant_port}
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
                kwargs["https"] = True
            client = QdrantClient(**kwargs)
            client.delete(
                collection_name=f"{tenant_id}_docs",
                points_selector=PointIdsList(points=point_ids),
            )
            deleted_chunks = chunk_count
            log.info("ingest.delete_chunks", tenant_id=tenant_id, doc_id=doc_id, chunks=deleted_chunks)

        # Delete file from local storage
        file_path = doc["path"]
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            log.info("ingest.delete_file", tenant_id=tenant_id, doc_id=doc_id, path=file_path)

        # Remove from Postgres
        await execute("DELETE FROM documents WHERE id=$1 AND tenant_id=$2", doc_id, tenant_id)

        log.info("ingest.deleted", tenant_id=tenant_id, doc_id=doc_id, chunks=deleted_chunks)
        return {
            "doc_id": doc_id,
            "filename": doc["filename"],
            "deleted_chunks": deleted_chunks,
            "status": "deleted",
        }

    except HTTPException:
        raise
    except Exception as exc:
        log.error("ingest.delete_failed", tenant_id=tenant_id, doc_id=doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Delete failed") from exc
    finally:
        log.debug("ingest.delete_exit", tenant_id=tenant_id, doc_id=doc_id)


_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB hard cap per upload


def _validate_file(file: UploadFile) -> None:
    try:
        ext = os.path.splitext(file.filename or "")[1].lower()
        if ext not in _ALLOWED:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type '{ext}'. Allowed: {sorted(_ALLOWED)}",
            )
        if file.size and file.size > _MAX_FILE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File too large ({file.size // (1024*1024)}MB). Max 50MB.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        log.error("ingest.validate_file_failed", filename=file.filename, error=str(exc))
        raise
    finally:
        log.debug("ingest.validate_file_exit", filename=file.filename)


async def _check_quota(tenant_id: str) -> None:
    try:
        tenant = await fetch_one("SELECT quota_docs FROM tenants WHERE id=$1", tenant_id)
        if not tenant:
            raise HTTPException(status_code=404, detail="Tenant not found")
        count = await tenant_doc_count(tenant_id)
        if count >= tenant["quota_docs"]:
            raise HTTPException(
                status_code=429,
                detail=f"Document quota reached ({tenant['quota_docs']})",
            )
    except HTTPException:
        raise
    except Exception as exc:
        log.error("ingest.check_quota_failed", tenant_id=tenant_id, error=str(exc))
        raise
    finally:
        log.debug("ingest.check_quota_exit", tenant_id=tenant_id)


async def _domain_queue(tenant_id: str, domain: str | None = None) -> str:
    try:
        if not domain:
            tenant = await fetch_one("SELECT domain FROM tenants WHERE id=$1", tenant_id)
            domain = tenant["domain"] if tenant else "general"
        queue_map = {
            "financial": "financial",
            "medical": "medical",
            "clinical": "medical",
            "legal": "legal",
            "scientific": "scientific",
            "technical": "technical",
        }
        return queue_map.get(domain, "general")
    except Exception as exc:
        log.warning("ingest.domain_queue_failed", tenant_id=tenant_id, error=str(exc))
        return "general"
    finally:
        log.debug("ingest.domain_queue_exit", tenant_id=tenant_id)
