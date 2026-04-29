"""Google OAuth 2.0 — /auth/google + /auth/google/callback + /auth/me."""
from __future__ import annotations

import re

import httpx
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, HnswConfigDiff, PayloadSchemaType,
                                  SparseIndexParams, SparseVectorParams,
                                  VectorParams)

from api.auth import get_tenant, issue_token
from config.logging import get_logger
from config.settings import settings
from db.postgres import execute, execute_returning, fetch_one
from ingestion.model_registry import MODEL_REGISTRY

log = get_logger()
router = APIRouter(prefix="/auth", tags=["auth"])

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v3/userinfo"


def _slug(email: str) -> str:
    """email → safe tenant_id slug: shobhitmohadikar@gmail.com → shobhitmohadikar."""
    local = email.split("@")[0]
    return re.sub(r"[^a-z0-9-]", "-", local.lower()).strip("-") or "user"


@router.get("/google")
async def google_login() -> RedirectResponse:
    """Redirect browser to Google consent screen."""
    if not settings.google_client_id:
        raise HTTPException(status_code=503, detail="Google OAuth not configured — set GOOGLE_CLIENT_ID")
    params = (
        f"client_id={settings.google_client_id}"
        f"&redirect_uri={settings.google_redirect_uri}"
        f"&response_type=code"
        f"&scope=openid+email+profile"
        f"&access_type=offline"
        f"&prompt=select_account"
    )
    return RedirectResponse(f"{_GOOGLE_AUTH_URL}?{params}")


@router.get("/google/callback")
async def google_callback(code: str) -> RedirectResponse:
    """Exchange code → user info → upsert tenant → issue JWT → redirect to frontend."""
    try:
        user = await _exchange_code(code)
        google_sub: str = user["sub"]
        email: str = user["email"]
        name: str = user.get("name", email)

        tenant_id = await _upsert_tenant(google_sub, email)
        token = issue_token(tenant_id)

        log.info("auth.google_login", tenant_id=tenant_id, email=email)
        return RedirectResponse(f"{settings.frontend_url}/callback?token={token}")

    except HTTPException:
        raise
    except Exception as exc:
        log.error("auth.google_callback_failed", error=str(exc))
        return RedirectResponse(f"{settings.frontend_url}/callback?error=auth_failed")


@router.get("/me")
async def me(tenant_id: str = Depends(get_tenant)) -> dict:
    """Return current tenant profile — requires Bearer token."""
    try:
        row = await fetch_one(
            "SELECT id, email, plan, domain, quota_docs, quota_tokens, created_at FROM tenants WHERE id=$1",
            tenant_id,
        )
        if not row:
            raise HTTPException(status_code=404, detail="Tenant not found")
        return {
            "tenant_id": row["id"],
            "email": row["email"],
            "plan": row["plan"],
            "domain": row["domain"],
            "quota_docs": row["quota_docs"],
            "quota_tokens": row["quota_tokens"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
        }
    except HTTPException:
        raise
    except Exception as exc:
        log.error("auth.me_failed", tenant_id=tenant_id, error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to fetch profile")


async def _exchange_code(code: str) -> dict:
    """Exchange OAuth code for user profile dict."""
    async with httpx.AsyncClient(timeout=10) as client:
        token_resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_client_id,
                "client_secret": settings.google_client_secret,
                "redirect_uri": settings.google_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        if token_resp.status_code != 200:
            log.error("auth.token_exchange_failed", status=token_resp.status_code, body=token_resp.text)
            raise HTTPException(status_code=401, detail="Google token exchange failed")

        access_token = token_resp.json().get("access_token")
        if not access_token:
            raise HTTPException(status_code=401, detail="No access token returned by Google")

        userinfo_resp = await client.get(
            _GOOGLE_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Failed to fetch Google user info")

        return userinfo_resp.json()


async def _upsert_tenant(google_sub: str, email: str) -> str:
    """Return existing tenant_id or create a new tenant. Links google_sub on first login."""

    # 1. Already linked by google_sub (returning user)
    row = await fetch_one("SELECT id FROM tenants WHERE google_sub=$1", google_sub)
    if row:
        return row["id"]

    # 2. Tenant exists by email but not yet linked (e.g. shobhitmohadikar seeded manually)
    row = await fetch_one("SELECT id FROM tenants WHERE email=$1", email)
    if row:
        await execute("UPDATE tenants SET google_sub=$1 WHERE id=$2", google_sub, row["id"])
        log.info("auth.google_sub_linked", tenant_id=row["id"], email=email)
        return row["id"]

    # 3. Brand-new user — provision tenant + default workspace
    tenant_id = _slug(email)
    # Handle collision: append a suffix
    suffix = 0
    base = tenant_id
    while await fetch_one("SELECT id FROM tenants WHERE id=$1", tenant_id):
        suffix += 1
        tenant_id = f"{base}{suffix}"

    spec = MODEL_REGISTRY["general"]

    await execute(
        """
        INSERT INTO tenants (id, plan, domain, embed_model, embed_dim, quota_docs, quota_tokens,
                             google_sub, email)
        VALUES ($1,'free','general',$2,$3,100,1000000,$4,$5)
        """,
        tenant_id, spec["model_id"], spec["dim"], google_sub, email,
    )

    # Auto-provision a default workspace + Qdrant collection
    await _provision_default_workspace(tenant_id, spec)

    log.info("auth.tenant_provisioned", tenant_id=tenant_id, email=email)
    return tenant_id


async def _provision_default_workspace(tenant_id: str, spec: dict) -> None:
    """Create Default workspace in Postgres + Qdrant collection."""
    row = await execute_returning(
        """
        INSERT INTO workspaces (tenant_id, name, domain, embed_model, embed_dim, collection_name)
        VALUES ($1,'Default','general',$2,$3,$4)
        RETURNING id
        """,
        tenant_id, spec["model_id"], spec["dim"], f"{tenant_id}_docs",
    )
    ws_id = row["id"]
    collection = f"{tenant_id}_docs"
    await execute("UPDATE workspaces SET collection_name=$1 WHERE id=$2", collection, ws_id)

    try:
        kwargs: dict = {"host": settings.qdrant_host, "port": settings.qdrant_port}
        if settings.qdrant_api_key:
            kwargs["api_key"] = settings.qdrant_api_key
            kwargs["https"] = True
        client = QdrantClient(**kwargs)
        client.create_collection(
            collection_name=collection,
            vectors_config={"dense": VectorParams(size=spec["dim"], distance=Distance.COSINE,
                                                   hnsw_config=HnswConfigDiff(m=16, ef_construct=200))},
            sparse_vectors_config={"sparse": SparseVectorParams(index=SparseIndexParams(full_scan_threshold=5000))},
        )
        client.create_payload_index(collection, "section_num", PayloadSchemaType.INTEGER)
        log.info("auth.qdrant_collection_created", tenant_id=tenant_id, collection=collection)
    except Exception as exc:
        log.warning("auth.qdrant_provision_failed", tenant_id=tenant_id, error=str(exc))
