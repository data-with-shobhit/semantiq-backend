CREATE TABLE IF NOT EXISTS tenants (
    id           TEXT PRIMARY KEY,
    plan         TEXT NOT NULL CHECK (plan IN ('free', 'pro', 'enterprise')),
    domain       TEXT NOT NULL CHECK (domain IN (
                     'general', 'financial', 'medical',
                     'clinical', 'legal', 'scientific', 'technical'
                 )),
    embed_model  TEXT NOT NULL,
    embed_dim    INTEGER NOT NULL CHECK (embed_dim > 0),
    quota_docs   INTEGER NOT NULL DEFAULT 100,
    quota_tokens BIGINT  NOT NULL DEFAULT 1000000,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- One tenant → many workspaces, each with its own domain + Qdrant collection
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
    collection_name TEXT        NOT NULL UNIQUE,   -- {tenant_id}_{id}_docs
    storage_bytes   BIGINT      NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (tenant_id, name)
);
CREATE INDEX IF NOT EXISTS idx_workspaces_tenant ON workspaces (tenant_id);

CREATE TABLE IF NOT EXISTS documents (
    id              SERIAL PRIMARY KEY,
    tenant_id       TEXT        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    workspace_id    INTEGER     NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    filename        TEXT        NOT NULL,
    path            TEXT        NOT NULL,
    status          TEXT        NOT NULL DEFAULT 'processing'
                        CHECK (status IN ('processing', 'ready', 'failed')),
    chunk_count     INTEGER,
    section_count   INTEGER     DEFAULT 0,
    avg_chunk_size  INTEGER     DEFAULT 0,
    file_size_bytes INTEGER     DEFAULT 0,
    error_msg       TEXT,
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_documents_tenant    ON documents (tenant_id);
CREATE INDEX IF NOT EXISTS idx_documents_workspace ON documents (workspace_id);
CREATE INDEX IF NOT EXISTS idx_documents_status    ON documents (tenant_id, status);

-- Stores chunking strategy per document, with version history for rollback
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
);
CREATE INDEX IF NOT EXISTS idx_strategies_doc    ON ingestion_strategies (doc_id);
CREATE INDEX IF NOT EXISTS idx_strategies_active ON ingestion_strategies (doc_id, is_active);
CREATE INDEX IF NOT EXISTS idx_strategies_name   ON ingestion_strategies (tenant_id, name) WHERE name IS NOT NULL;

-- RAGAS evaluation runs — one row per eval session
CREATE TABLE IF NOT EXISTS ragas_evals (
    id              SERIAL PRIMARY KEY,
    doc_id          INTEGER     NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id       TEXT        NOT NULL REFERENCES tenants(id)   ON DELETE CASCADE,
    strategy_id     INTEGER     REFERENCES ingestion_strategies(id) ON DELETE SET NULL,
    questions       JSONB       NOT NULL,
    ground_truths   JSONB       NOT NULL,
    answers         JSONB,
    contexts        JSONB,
    faithfulness    FLOAT,
    context_precision FLOAT,
    answer_relevance FLOAT,
    avg_score       FLOAT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ragas_doc ON ragas_evals (doc_id, created_at DESC);

CREATE TABLE IF NOT EXISTS usage (
    id          SERIAL PRIMARY KEY,
    tenant_id   TEXT        NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    event_type  TEXT        NOT NULL CHECK (event_type IN ('ingest', 'query')),
    tokens_used INTEGER,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_usage_tenant_time ON usage (tenant_id, occurred_at);
