"""Integration tests for db/postgres.py — requires live Postgres."""
import os

import pytest
import pytest_asyncio

from db.postgres import (close_pool, create_document, execute, fetch_one,
                         get_tenant, init_pool, mark_document_failed,
                         mark_document_ready, record_usage, tenant_doc_count)

os.environ.setdefault("DATABASE_URL", "postgresql://raguser:ragpass@localhost:5432/ragdb")


TENANT_ID = "__test_tenant__"


@pytest_asyncio.fixture(autouse=True)
async def pool():
    await init_pool()
    yield
    await close_pool()


@pytest_asyncio.fixture(autouse=True)
async def clean(pool):
    await execute("DELETE FROM tenants WHERE id = $1", TENANT_ID)
    yield
    await execute("DELETE FROM tenants WHERE id = $1", TENANT_ID)


async def _insert_tenant() -> None:
    await execute(
        """
        INSERT INTO tenants (id, plan, domain, embed_model, embed_dim, quota_docs, quota_tokens)
        VALUES ($1, 'free', 'general', 'BAAI/bge-m3', 1024, 100, 1000000)
        ON CONFLICT DO NOTHING
        """,
        TENANT_ID,
    )


class TestSchema:
    async def test_tables_exist(self) -> None:
        for table in ("tenants", "documents", "usage"):
            row = await fetch_one(
                "SELECT to_regclass($1) AS t",
                f"public.{table}",
            )
            assert row["t"] is not None, f"Table '{table}' missing"

    async def test_tenants_constraints(self) -> None:
        with pytest.raises(Exception):
            await execute(
                "INSERT INTO tenants (id, plan, domain, embed_model, embed_dim, quota_docs, quota_tokens)"
                " VALUES ('bad', 'unknown_plan', 'general', 'x', 768, 100, 1000000)"
            )

    async def test_documents_status_constraint(self) -> None:
        await _insert_tenant()
        with pytest.raises(Exception):
            await execute(
                "INSERT INTO documents (tenant_id, filename, path, status)"
                " VALUES ($1, 'f.pdf', '/f.pdf', 'invalid_status')",
                TENANT_ID,
            )


class TestTenantHelpers:
    async def test_get_tenant_missing(self) -> None:
        row = await get_tenant("does-not-exist")
        assert row is None

    async def test_get_tenant_found(self) -> None:
        await _insert_tenant()
        row = await get_tenant(TENANT_ID)
        assert row is not None
        assert row["domain"] == "general"
        assert row["embed_dim"] == 1024

    async def test_tenant_doc_count_zero(self) -> None:
        await _insert_tenant()
        assert await tenant_doc_count(TENANT_ID) == 0


class TestDocumentHelpers:
    async def test_create_and_mark_ready(self) -> None:
        await _insert_tenant()
        doc_id = await create_document(TENANT_ID, "test.pdf", "/storage/test.pdf")
        assert isinstance(doc_id, int)

        row = await fetch_one("SELECT status FROM documents WHERE id = $1", doc_id)
        assert row["status"] == "processing"

        await mark_document_ready(doc_id, 42, TENANT_ID)
        row = await fetch_one("SELECT status, chunk_count FROM documents WHERE id = $1", doc_id)
        assert row["status"] == "ready"
        assert row["chunk_count"] == 42

    async def test_mark_failed(self) -> None:
        await _insert_tenant()
        doc_id = await create_document(TENANT_ID, "bad.pdf", "/storage/bad.pdf")
        await mark_document_failed(doc_id, "parse error", TENANT_ID)
        row = await fetch_one("SELECT status, error_msg FROM documents WHERE id = $1", doc_id)
        assert row["status"] == "failed"
        assert row["error_msg"] == "parse error"


class TestUsageHelpers:
    async def test_record_usage(self) -> None:
        await _insert_tenant()
        await record_usage(TENANT_ID, "ingest", 500)
        row = await fetch_one(
            "SELECT tokens_used FROM usage WHERE tenant_id=$1 AND event_type='ingest'",
            TENANT_ID,
        )
        assert row["tokens_used"] == 500

    async def test_usage_cascade_delete(self) -> None:
        await _insert_tenant()
        await record_usage(TENANT_ID, "query", 100)
        await execute("DELETE FROM tenants WHERE id=$1", TENANT_ID)
        row = await fetch_one(
            "SELECT COUNT(*) AS n FROM usage WHERE tenant_id=$1", TENANT_ID
        )
        assert row["n"] == 0
