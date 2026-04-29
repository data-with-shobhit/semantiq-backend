"""Workspace CRUD — one tenant, many isolated domain workspaces.

POST   /workspaces              — create new workspace + Qdrant collection
GET    /workspaces              — list all workspaces with storage usage
DELETE /workspaces/{id}         — delete workspace + docs + collection + files
GET    /workspaces/storage      — tenant storage summary (used / 200MB quota)
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, SparseIndexParams,
                                  SparseVectorParams, VectorParams)

from api.auth import get_tenant
from api.models.workspaces import WorkspaceCreate
from config.logging import get_logger
from config.settings import settings
from db.postgres import (_STORAGE_QUOTA_BYTES, create_workspace, execute,
                         fetch_all, get_workspace, list_workspaces,
                         tenant_storage_bytes)
from ingestion.model_registry import MODEL_REGISTRY
from storage.r2 import R2Storage

log = get_logger()
router = APIRouter(prefix="/workspaces", tags=["workspaces"])

_MB = 1024 * 1024


def _qdrant() -> QdrantClient:
    kwargs: dict = {"host": settings.qdrant_host, "port": settings.qdrant_port, "timeout": 30}
    if settings.qdrant_api_key:
        kwargs["api_key"] = settings.qdrant_api_key
        kwargs["https"] = True
    return QdrantClient(**kwargs)



def _fmt(row: Any, storage_bytes: int | None = None) -> dict:
    sb = storage_bytes if storage_bytes is not None else (row["storage_bytes"] or 0)
    return {
        "id": row["id"],
        "name": row["name"],
        "domain": row["domain"],
        "embed_model": row["embed_model"],
        "embed_dim": row["embed_dim"],
        "collection_name": row["collection_name"],
        "storage_mb": round(sb / _MB, 2),
        "created_at": row["created_at"].isoformat() if row["created_at"] else None,
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create(
    req: WorkspaceCreate,
    tenant_id: str = Depends(get_tenant),
) -> dict:
    model_info = MODEL_REGISTRY[req.domain]
    embed_model = model_info["model_id"]
    embed_dim = model_info["dim"]

    try:
        ws = await create_workspace(tenant_id, req.name, req.domain, embed_model, embed_dim)
    except Exception as exc:
        if "unique" in str(exc).lower():
            raise HTTPException(status_code=409, detail=f"Workspace '{req.name}' already exists")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Create Qdrant collection for this workspace
    try:
        client = _qdrant()
        client.create_collection(
            collection_name=ws["collection_name"],
            vectors_config={
                "dense": VectorParams(size=embed_dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
            },
        )
        # Payload index for section-based direct lookup
        client.create_payload_index(
            collection_name=ws["collection_name"],
            field_name="section_num",
            field_schema="integer",
        )
        log.info("workspace.collection_created", tenant_id=tenant_id,
                 collection=ws["collection_name"], dim=embed_dim)
    except Exception as exc:
        # Roll back workspace row if Qdrant fails
        await execute("DELETE FROM workspaces WHERE id=$1", ws["id"])
        log.error("workspace.qdrant_failed", tenant_id=tenant_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Qdrant collection creation failed: {exc}") from exc

    log.info("workspace.created", tenant_id=tenant_id, workspace_id=ws["id"], name=req.name, domain=req.domain)
    return _fmt(ws)


@router.get("", status_code=status.HTTP_200_OK)
async def list_all(
    tenant_id: str = Depends(get_tenant),
) -> dict:
    rows = await list_workspaces(tenant_id)
    used_bytes = await tenant_storage_bytes(tenant_id)
    return {
        "tenant_id": tenant_id,
        "storage_used_mb": round(used_bytes / _MB, 2),
        "storage_quota_mb": round(_STORAGE_QUOTA_BYTES / _MB, 2),
        "storage_remaining_mb": round((_STORAGE_QUOTA_BYTES - used_bytes) / _MB, 2),
        "workspaces": [_fmt(r) for r in rows],
    }


@router.get("/storage", status_code=status.HTTP_200_OK)
async def storage_summary(
    tenant_id: str = Depends(get_tenant),
) -> dict:
    used = await tenant_storage_bytes(tenant_id)
    rows = await list_workspaces(tenant_id)
    return {
        "tenant_id": tenant_id,
        "storage_used_mb": round(used / _MB, 2),
        "storage_quota_mb": round(_STORAGE_QUOTA_BYTES / _MB, 2),
        "storage_remaining_mb": round((_STORAGE_QUOTA_BYTES - used) / _MB, 2),
        "pct_used": round(used / _STORAGE_QUOTA_BYTES * 100, 1),
        "workspaces": [
            {"id": r["id"], "name": r["name"], "storage_mb": round((r["storage_bytes"] or 0) / _MB, 2)}
            for r in rows
        ],
    }


@router.delete("/{workspace_id}", status_code=status.HTTP_200_OK)
async def delete(
    workspace_id: int,
    tenant_id: str = Depends(get_tenant),
) -> dict:
    ws = await get_workspace(workspace_id, tenant_id)
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Delete all files in workspace
    docs = await fetch_all(
        "SELECT path, file_size_bytes FROM documents WHERE workspace_id=$1 AND tenant_id=$2",
        workspace_id, tenant_id,
    )
    deleted_files = 0
    for doc in docs:
        path = doc["path"]
        if path and not path.startswith("r2://") and os.path.exists(path):
            try:
                os.remove(path)
                deleted_files += 1
            except Exception:
                pass
        elif path and path.startswith("r2://"):
            try:
                R2Storage().delete(path)
                deleted_files += 1
            except Exception:
                pass

    # Delete Qdrant collection
    try:
        client = _qdrant()
        client.delete_collection(ws["collection_name"])
        log.info("workspace.collection_deleted", tenant_id=tenant_id, collection=ws["collection_name"])
    except Exception as exc:
        log.warning("workspace.qdrant_delete_failed", collection=ws["collection_name"], error=str(exc))

    # Cascade delete from Postgres (documents + strategies + evals via FK)
    await execute("DELETE FROM workspaces WHERE id=$1 AND tenant_id=$2", workspace_id, tenant_id)

    log.info("workspace.deleted", tenant_id=tenant_id, workspace_id=workspace_id,
             name=ws["name"], files_deleted=deleted_files)
    return {
        "workspace_id": workspace_id,
        "name": ws["name"],
        "files_deleted": deleted_files,
        "storage_freed_mb": round((ws["storage_bytes"] or 0) / _MB, 2),
        "status": "deleted",
    }
