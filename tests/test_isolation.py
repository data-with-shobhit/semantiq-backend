"""CRITICAL: Cross-tenant isolation tests.

Any failure here = build failure. A tenant must NEVER access another tenant's data.
Requires live Postgres + Qdrant + Redis.
"""
from __future__ import annotations

import os

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from api.auth import issue_token
from api.main import app
from cache.redis_client import cache_set, flush_tenant, get_client
from db.postgres import close_pool, execute, fetch_all, init_pool

os.environ.setdefault("DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("QDRANT_PORT", "6333")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")



_TENANT_A = "__isolation_tenant_a__"
_TENANT_B = "__isolation_tenant_b__"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest_asyncio.fixture(autouse=True)
async def setup_teardown():
    await init_pool()
    for t in (_TENANT_A, _TENANT_B):
        await execute("DELETE FROM tenants WHERE id=$1", t)
    yield
    for t in (_TENANT_A, _TENANT_B):
        await execute("DELETE FROM tenants WHERE id=$1", t)
        flush_tenant(t)
        _drop_qdrant_collection(f"{t}_docs")
    await close_pool()


def _drop_qdrant_collection(name: str) -> None:
    try:
        QdrantClient(host="localhost", port=6333).delete_collection(name)
    except Exception:
        pass


def _create_qdrant_collection(name: str, dim: int = 768) -> None:
    QdrantClient(host="localhost", port=6333).recreate_collection(
        collection_name=name,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


class TestJWTIsolation:
    """Tenant A's JWT must not grant access to Tenant B's resources."""

    async def test_token_sub_matches_tenant(self) -> None:
        token_a = issue_token(_TENANT_A)
        from api.auth import get_tenant
        tenant_id = await get_tenant(token_a)
        assert tenant_id == _TENANT_A
        assert tenant_id != _TENANT_B

    async def test_cross_tenant_token_rejected(self) -> None:
        import jwt as pyjwt
        from fastapi import HTTPException
        tampered = pyjwt.encode({"sub": _TENANT_B}, "wrong-secret", algorithm="HS256")
        from api.auth import get_tenant
        with pytest.raises(HTTPException) as exc:
            await get_tenant(tampered)
        assert exc.value.status_code == 401


class TestPostgresIsolation:
    """Postgres queries must always be scoped to the correct tenant_id."""

    async def test_documents_scoped_to_tenant(self) -> None:
        await execute(
            "INSERT INTO tenants (id,plan,domain,embed_model,embed_dim,quota_docs,quota_tokens)"
            " VALUES ($1,'free','general','BAAI/bge-m3',1024,100,1000000)",
            _TENANT_A,
        )
        await execute(
            "INSERT INTO tenants (id,plan,domain,embed_model,embed_dim,quota_docs,quota_tokens)"
            " VALUES ($1,'free','general','BAAI/bge-m3',1024,100,1000000)",
            _TENANT_B,
        )
        await execute(
            "INSERT INTO documents (tenant_id,filename,path,status) VALUES ($1,'a.pdf','/a','ready')",
            _TENANT_A,
        )
        rows = await fetch_all("SELECT * FROM documents WHERE tenant_id=$1", _TENANT_B)
        assert len(rows) == 0, "Tenant B must not see Tenant A's documents"

    async def test_usage_scoped_to_tenant(self) -> None:
        await execute(
            "INSERT INTO tenants (id,plan,domain,embed_model,embed_dim,quota_docs,quota_tokens)"
            " VALUES ($1,'free','general','BAAI/bge-m3',1024,100,1000000) ON CONFLICT DO NOTHING",
            _TENANT_A,
        )
        await execute(
            "INSERT INTO usage (tenant_id,event_type,tokens_used) VALUES ($1,'query',100)",
            _TENANT_A,
        )
        rows = await fetch_all("SELECT * FROM usage WHERE tenant_id=$1", _TENANT_B)
        assert len(rows) == 0, "Tenant B must not see Tenant A's usage"


class TestQdrantIsolation:
    """Vector search must be locked to the tenant's own collection."""

    def test_collection_names_are_distinct(self) -> None:
        assert f"{_TENANT_A}_docs" != f"{_TENANT_B}_docs"

    def test_search_uses_tenant_collection(self) -> None:
        _create_qdrant_collection(f"{_TENANT_A}_docs")
        _create_qdrant_collection(f"{_TENANT_B}_docs")
        client = QdrantClient(host="localhost", port=6333)
        names = [c.name for c in client.get_collections().collections]
        assert f"{_TENANT_A}_docs" in names
        assert f"{_TENANT_B}_docs" in names
        assert f"{_TENANT_A}_docs" != f"{_TENANT_B}_docs"


class TestRedisIsolation:
    """Redis keys must be prefixed with tenant_id — no cross-tenant reads."""

    def test_cache_keys_are_prefixed(self) -> None:
        cache_set(_TENANT_A, "what is revenue?", "Revenue is money earned.")
        cache_set(_TENANT_B, "what is revenue?", "Different answer.")
        r = get_client()
        keys_a = r.keys(f"{_TENANT_A}:*")
        keys_b = r.keys(f"{_TENANT_B}:*")
        assert all(_TENANT_A in k for k in keys_a)
        assert all(_TENANT_B in k for k in keys_b)
        assert not any(_TENANT_B in k for k in keys_a)
        assert not any(_TENANT_A in k for k in keys_b)

    def test_flush_tenant_only_deletes_own_keys(self) -> None:
        cache_set(_TENANT_A, "q1", "answer a")
        cache_set(_TENANT_B, "q1", "answer b")
        flush_tenant(_TENANT_A)
        r = get_client()
        assert len(r.keys(f"{_TENANT_A}:*")) == 0
        assert len(r.keys(f"{_TENANT_B}:*")) > 0


class TestStorageIsolation:
    """File paths must be scoped to storage/docs/{tenant_id}/."""

    def test_storage_paths_are_isolated(self) -> None:
        import tempfile

        from storage.local import LocalStorage
        with tempfile.TemporaryDirectory() as tmp:
            storage = LocalStorage(root=tmp)
            path_a = storage.save(_TENANT_A, "secret.pdf", b"tenant a data")
            path_b = storage.save(_TENANT_B, "secret.pdf", b"tenant b data")
            assert path_a != path_b
            assert _TENANT_A in path_a
            assert _TENANT_A not in path_b
            assert storage.load(path_a) != storage.load(path_b)
