"""
Idempotent migration script — run once to bring an existing DB up to the
current schema.

Usage:
    uv run python -m db.migrate
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import asyncpg

from config.logging import configure_logging, get_logger
from config.settings import settings
from ingestion.model_registry import MODEL_REGISTRY

sys.path.insert(0, str(Path(__file__).parent.parent))



configure_logging(settings.log_level, settings.log_format)
log = get_logger()


async def run() -> None:
    conn = await asyncpg.connect(settings.database_url)
    try:
        await _migrate(conn)
    finally:
        await conn.close()


async def _migrate(conn: asyncpg.Connection) -> None:
    log.info("migrate.start")

    # ── 1. workspaces table ───────────────────────────────────────────────────
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS workspaces (
            id              SERIAL PRIMARY KEY,
            tenant_id       TEXT        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            name            TEXT        NOT NULL,
            domain          TEXT        NOT NULL CHECK (domain IN (
                                'general', 'financial', 'medical',
                                'clinical', 'legal', 'scientific', 'technical'
                            )),
            embed_model     TEXT        NOT NULL,
            embed_dim       INTEGER     NOT NULL CHECK (embed_dim > 0),
            collection_name TEXT        NOT NULL UNIQUE,
            storage_bytes   BIGINT      NOT NULL DEFAULT 0,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            UNIQUE (tenant_id, name)
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_workspaces_tenant ON workspaces (tenant_id)"
    )
    log.info("migrate.workspaces_table_ok")

    # ── 2. workspace_id column on documents (nullable — filled below) ─────────
    col_exists = await conn.fetchval("""
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_name='documents' AND column_name='workspace_id'
    """)
    if not col_exists:
        await conn.execute(
            "ALTER TABLE documents ADD COLUMN workspace_id INTEGER REFERENCES workspaces(id) ON DELETE CASCADE"
        )
        log.info("migrate.workspace_id_column_added")
    else:
        log.info("migrate.workspace_id_column_exists")

    # ── 3. file_size_bytes column on documents ────────────────────────────────
    size_exists = await conn.fetchval("""
        SELECT COUNT(*) FROM information_schema.columns
        WHERE table_name='documents' AND column_name='file_size_bytes'
    """)
    if not size_exists:
        await conn.execute(
            "ALTER TABLE documents ADD COLUMN file_size_bytes INTEGER DEFAULT 0"
        )
        log.info("migrate.file_size_bytes_column_added")

    # ── 4. ingestion_strategies table ────────────────────────────────────────
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS ingestion_strategies (
            id              SERIAL PRIMARY KEY,
            doc_id          INTEGER     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            tenant_id       TEXT        NOT NULL REFERENCES tenants(id)   ON DELETE CASCADE,
            version         INTEGER     NOT NULL DEFAULT 1,
            chunker         TEXT        NOT NULL,
            chunk_size      INTEGER     NOT NULL,
            overlap         INTEGER     NOT NULL DEFAULT 0,
            extra_params    JSONB       NOT NULL DEFAULT '{}',
            reasoning       TEXT,
            source          TEXT        NOT NULL DEFAULT 'heuristic'
                                CHECK (source IN ('heuristic', 'llm', 'llm_ragas', 'user_revert')),
            ragas_scores    JSONB,
            user_feedback   TEXT,
            name            TEXT,
            is_active       BOOLEAN     NOT NULL DEFAULT TRUE,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strategies_doc ON ingestion_strategies (doc_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strategies_active ON ingestion_strategies (doc_id, is_active)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_strategies_name ON ingestion_strategies (tenant_id, name) WHERE name IS NOT NULL"
    )
    log.info("migrate.ingestion_strategies_table_ok")

    # ── 5. ragas_evals table ──────────────────────────────────────────────────
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS ragas_evals (
            id                SERIAL PRIMARY KEY,
            doc_id            INTEGER     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            tenant_id         TEXT        NOT NULL REFERENCES tenants(id)   ON DELETE CASCADE,
            strategy_id       INTEGER     REFERENCES ingestion_strategies(id) ON DELETE SET NULL,
            questions         JSONB       NOT NULL,
            ground_truths     JSONB       NOT NULL,
            answers           JSONB,
            contexts          JSONB,
            faithfulness      FLOAT,
            context_precision FLOAT,
            answer_relevance  FLOAT,
            avg_score         FLOAT,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_ragas_doc ON ragas_evals (doc_id, created_at DESC)"
    )
    log.info("migrate.ragas_evals_table_ok")

    # ── 6. Create default workspace for every tenant that has none ────────────
    tenants = await conn.fetch("SELECT id, domain, embed_model, embed_dim FROM tenants")
    log.info("migrate.tenants_found", count=len(tenants))

    for tenant in tenants:
        tid = tenant["id"]
        domain = tenant["domain"] or "general"
        model_info = MODEL_REGISTRY.get(domain, MODEL_REGISTRY["general"])
        embed_model = tenant["embed_model"] or model_info["model_id"]
        embed_dim = tenant["embed_dim"] or model_info["dim"]

        ws_count = await conn.fetchval(
            "SELECT COUNT(*) FROM workspaces WHERE tenant_id=$1", tid
        )
        if ws_count > 0:
            log.info("migrate.tenant_has_workspace", tenant_id=tid, workspaces=ws_count)
            continue

        # Legacy collection name is {tenant_id}_docs
        legacy_collection = f"{tid}_docs"

        # Check if legacy collection name is already taken by another workspace
        taken = await conn.fetchval(
            "SELECT COUNT(*) FROM workspaces WHERE collection_name=$1", legacy_collection
        )
        collection_name = legacy_collection if not taken else f"{tid}_default_docs"

        ws_id = await conn.fetchval(
            """
            INSERT INTO workspaces (tenant_id, name, domain, embed_model, embed_dim, collection_name)
            VALUES ($1, 'Default', $2, $3, $4, $5)
            RETURNING id
            """,
            tid, domain, embed_model, embed_dim, collection_name,
        )
        log.info("migrate.workspace_created", tenant_id=tid, workspace_id=ws_id, collection=collection_name)

        # Assign orphaned documents to this workspace
        updated = await conn.execute(
            "UPDATE documents SET workspace_id=$1 WHERE tenant_id=$2 AND workspace_id IS NULL",
            ws_id, tid,
        )
        log.info("migrate.docs_assigned", tenant_id=tid, workspace_id=ws_id, result=updated)

        # Sync storage_bytes from sum of existing doc sizes
        total = await conn.fetchval(
            "SELECT COALESCE(SUM(file_size_bytes), 0) FROM documents WHERE workspace_id=$1",
            ws_id,
        )
        await conn.execute(
            "UPDATE workspaces SET storage_bytes=$1 WHERE id=$2", total, ws_id
        )
        log.info("migrate.storage_synced", tenant_id=tid, workspace_id=ws_id, bytes=total)

    # ── 7. Add document indexes if missing ───────────────────────────────────
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_tenant    ON documents (tenant_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents (workspace_id)"
    )
    await conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_documents_status    ON documents (tenant_id, status)"
    )

    log.info("migrate.done")
    print("\nMigration complete.")
    print("Tenants migrated:")
    for t in tenants:
        ws = await conn.fetchrow(
            "SELECT id, name, collection_name FROM workspaces WHERE tenant_id=$1 ORDER BY id LIMIT 1",
            t["id"],
        )
        docs = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE tenant_id=$1", t["id"]
        )
        if ws:
            print(f"  {t['id']:30s}  workspace={ws['id']}  collection={ws['collection_name']}  docs={docs}")


if __name__ == "__main__":
    asyncio.run(run())
