"""POST /signup — provision a new tenant."""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException, status
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, PayloadSchemaType,
                                  SparseIndexParams, SparseVectorParams,
                                  VectorParams)

from api.auth import issue_token
from api.models.tenants import SignupRequest, SignupResponse
from config.logging import get_logger
from config.settings import settings
from db.postgres import execute, execute_returning, fetch_one
from ingestion.model_registry import get_model_spec

log = get_logger()
router = APIRouter(prefix="/tenants", tags=["tenants"])

def _qdrant() -> QdrantClient:
    try:
        kwargs: dict = {"host": settings.qdrant_host, "port": settings.qdrant_port}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
            kwargs["https"] = True
        return QdrantClient(**kwargs)
    except Exception as exc:
        log.error("tenants.qdrant_connect_failed", error=str(exc))
        raise



@router.post("/signup", response_model=SignupResponse, status_code=status.HTTP_201_CREATED)
async def signup(req: SignupRequest) -> SignupResponse:
    try:
        existing = await fetch_one("SELECT id FROM tenants WHERE id = $1", req.tenant_id)
        if existing:
            raise HTTPException(status_code=409, detail="Tenant already exists")

        spec = get_model_spec(req.domain)
        quota_docs = 500 if req.plan == "pro" else (2000 if req.plan == "enterprise" else 100)
        quota_tokens = 10_000_000 if req.plan != "free" else 1_000_000

        await execute_returning(
            """
            INSERT INTO tenants (id, plan, domain, embed_model, embed_dim, quota_docs, quota_tokens, search_context, web_search_enabled)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            RETURNING id
            """,
            req.tenant_id, req.plan, req.domain,
            spec["model_id"], spec["dim"], quota_docs, quota_tokens, req.search_context, req.web_search_enabled,
        )

        collection = f"{req.tenant_id}_docs"
        try:
            client = _qdrant()
            if client.collection_exists(collection):
                client.delete_collection(collection)
            client.create_collection(
                collection_name=collection,
                vectors_config={"dense": VectorParams(size=spec["dim"], distance=Distance.COSINE)},
                sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))},
            )
            # Payload indexes for fast filtering
            for field, schema in [
                ("doc_id",       PayloadSchemaType.INTEGER),
                ("chunk_index",  PayloadSchemaType.INTEGER),
                ("section_num",  PayloadSchemaType.INTEGER),
                ("section",      PayloadSchemaType.KEYWORD),
                ("domain",       PayloadSchemaType.KEYWORD),
                ("tenant_id",    PayloadSchemaType.KEYWORD),
            ]:
                client.create_payload_index(collection, field, field_schema=schema)
            os.makedirs(f"{settings.storage_root}/{req.tenant_id}", exist_ok=True)
        except Exception as exc:
            await execute("DELETE FROM tenants WHERE id = $1", req.tenant_id)
            log.error("signup.rollback", tenant_id=req.tenant_id, error=str(exc))
            raise HTTPException(status_code=500, detail="Provisioning failed; rolled back") from exc

        token = issue_token(req.tenant_id)
        log.info("tenant.created", tenant_id=req.tenant_id, domain=req.domain, plan=req.plan)
        return SignupResponse(
            tenant_id=req.tenant_id,
            token=token,
            embed_model=spec["model_id"],
            embed_dim=spec["dim"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        log.error("signup.unexpected_error", tenant_id=req.tenant_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Internal server error during signup") from exc
    finally:
        log.debug("signup.exit", tenant_id=req.tenant_id)
