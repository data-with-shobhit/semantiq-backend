# Semantiq — Project Context

This file is loaded at the start of every Claude Code session. It tells Claude
what this project is, what decisions have been made, and what conventions to follow.

---

## What this project is

**Semantiq** — a production-grade, multi-tenant CRAG (Corrective Retrieval-Augmented Generation) platform. Multiple tenants each get logically isolated document stores on shared infrastructure. Like Google Drive for RAG — one platform, completely separate data per tenant.

Each tenant signs up via Google OAuth, creates workspaces by domain, uploads documents, and receives answers strictly grounded in their own data via SSE-streamed chat.

See `ARCHITECTURE.md` for the full technical design and decision rationale.

---

## Current tech stack

**API + auth**
- FastAPI (async Python gateway)
- PyJWT (JWT signing / verify)
- slowapi (per-tenant rate limiting)
- Google OAuth 2.0 (sign in flow — backend-driven, issues JWT on callback)

**LLM serving**
- Groq `llama-3.3-70b-versatile` — primary generation backend
- Gemini 2.0 Flash — HyDE rewriter (falls back to Groq on 429/error)
- OpenRouter `google/gemma-4-31b-it:free` — optional fallback
- `LLM_PROVIDER` env var switches backends: `groq` (default) | `lmstudio` | `anthropic`
- LMBackend abstraction — never wire LLM calls directly in routes

**Orchestration**
- LangGraph CRAG agent (not plain ReAct — includes web search fallback on low retrieval score)
- No LangChain — direct libraries only

**Embeddings — Voyage AI (API, no local models)**
- `voyage-4` — general, medical, clinical, scientific (512-dim)
- `voyage-finance-2` — financial (1024-dim)
- `voyage-law-2` — legal (1024-dim)
- `voyage-code-3` — technical (1024-dim)
- `voyage rerank-2.5` — reranker (replaces CrossEncoder)

**Retrieval**
- SPLADE `naver/splade-v3` (learned sparse) + `rank-bm25` fallback
- Hybrid RRF fusion (k=60)
- Score threshold: 0.65 (below → Tavily web search → generate)
- Max 5 chunks in LLM prompt
- History trimmed to last 4 turns

**Storage (all cloud)**
- Qdrant Cloud — vector DB, one collection per workspace
- Supabase (Postgres 16) — tenant registry, workspaces, documents, eval results
- Upstash Redis — semantic cache + Celery broker/backend (TLS)
- Cloudflare R2 — uploaded files at `docs/{tenant_id}/{filename}`

**Infra + observability**
- Celery (async ingestion via Upstash broker)
- Docker Compose — local dev only
- LangSmith (LangGraph traces automatically via `LANGCHAIN_TRACING_V2`)
- Prometheus + Grafana (metrics + dashboards)
- RAGAS (faithfulness, context precision, answer relevancy — Groq LLM + Voyage embeddings)
- Tavily — web search fallback in CRAG agent

**Frontend**
- Next.js 14 App Router, TypeScript, Tailwind CSS
- Zustand (auth token, localStorage persist)
- TanStack Query (server state, polling)
- SSE streaming chat
- Deployed: Vercel → `semantiq-frontend-henna.vercel.app`
- Repo: `github.com/data-with-shobhit/semantiq-frontend-`

**Framework philosophy — no LangChain.** Direct libraries for every step. LangGraph standalone for CRAG agent. If the project ever needs 10+ document source types, revisit.

---

## Multi-tenancy model (non-negotiable)

- `tenant_id` is **always** extracted server-side from the JWT `sub` claim.
  Never accept `tenant_id` from user input, URL params, or request bodies.
- Qdrant collection name: `{tenant_id}_{workspace_id}_docs` — computed server-side only.
- Every Postgres query includes `WHERE tenant_id = $1`.
- Every Redis key is prefixed `{tenant_id}:`.
- Every R2 path is `docs/{tenant_id}/`.
- A tenant **cannot** reach another tenant's data through any code path.
- `test_isolation.py` runs on every commit — cross-tenant leak = build failure.

---

## Embedding strategy

Per-workspace domain selection. Model chosen at workspace creation, locked for the life of the collection. **Never mix models within a collection.**

| Domain | Voyage Model | Dim |
|---|---|---|
| general (default) | `voyage-4` | 512 |
| financial | `voyage-finance-2` | 1024 |
| legal | `voyage-law-2` | 1024 |
| medical | `voyage-4` | 512 |
| clinical | `voyage-4` | 512 |
| scientific | `voyage-4` | 512 |
| technical | `voyage-code-3` | 1024 |

Rules:
- Qdrant collection vector dim MUST match model dim.
- Model registry is single source of truth: `ingestion/model_registry.py`.
- Use `model_id` key (not `hf_id`) when reading from registry.
- Voyage embeddings are API calls — no local model loading.

---

## Context engineering rules

- HyDE rewrite: Gemini 2.0 Flash primary → Groq fallback on any error.
- SPLADE handles vocabulary expansion — no manual synonym dicts.
- Score threshold: 0.65. Below → Tavily web search. No result → "I don't have that information in your documents."
- Max 5 chunks in LLM prompt.
- Trim conversation history to last 4 turns.
- System prompt: every claim must be in sources. No source labels in answers (no "DOC Source 1"). No inference from training knowledge.
- Streaming: `/query/stream` emits SSE `{type: status|token|done}` events.

---

## Project structure

```
api/
├── main.py              FastAPI app entrypoint + CORS + lifespan
├── auth.py              JWT middleware, get_tenant() dependency
├── deps.py              shared FastAPI dependencies
└── routes/
    ├── auth.py          Google OAuth flow + tenant auto-provisioning
    ├── workspaces.py    Workspace CRUD + Qdrant collection provisioning
    ├── ingest.py        File upload → R2 → Celery enqueue
    ├── query.py         CRAG query + SSE streaming
    └── eval.py          RAGAS eval + strategy management

ingestion/
├── model_registry.py    domain → Voyage model mapping (source of truth)
├── embedder.py          Voyage AI embedding client
├── chunker.py           SentenceSplitter (default) + tree-sitter AST (technical)
├── loader.py            pypdf + unstructured wrappers
├── extractor.py         metadata extraction per chunk
└── tasks.py             Celery ingestion task

retrieval/
├── rewriter.py          HyDE (Gemini primary → Groq fallback)
├── splade.py            SPLADE naver/splade-v3 + BM25 fallback
├── hybrid.py            SPLADE + Voyage dense + RRF fusion
└── reranker.py          Voyage rerank-2.5

graph/
└── query_graph.py       LangGraph CRAG agent (retrieve → rerank → threshold → generate/web)

llm/
├── backend.py           LLMBackend protocol
├── groq_backend.py      Groq (default)
└── anthropic_backend.py optional Claude

storage/
├── backend.py           StorageBackend protocol
└── r2.py                Cloudflare R2 implementation

db/
├── postgres.py          asyncpg connection + query helpers
└── schema.sql           tenants, workspaces, documents, eval_results, eval_strategies

cache/
└── redis_client.py      semantic cache + tenant-prefixed keys (Upstash)

eval/
└── ragas_eval.py        RAGAS harness (Groq key rotation, Voyage embeddings)

frontend/                Next.js 14 frontend (separate git repo)

tests/
├── test_auth.py
├── test_isolation.py    CRITICAL: cross-tenant leak test
└── test_retrieval.py

monitoring/
├── prometheus.yml
└── grafana_dashboard.json

Dockerfile               Cloud Run deployment image
.dockerignore
docker-compose.yml       local dev only (Postgres + Redis + Qdrant)
requirements.txt
CLAUDE.md                this file
ARCHITECTURE.md          deep technical design
```

---

## Coding conventions

- Python 3.11+, type hints on all function signatures.
- FastAPI routes use `tenant_id: str = Depends(get_tenant)` — never parse JWT manually.
- No `print()` — use `structlog`. Every log line includes `tenant_id`.
- Async everywhere in API layer. Celery workers are sync.
- Functions ≤ 50 lines. Split if longer.
- Test files mirror source tree (`api/auth.py` → `tests/test_auth.py`).
- Read from model registry — never hardcode Voyage model IDs in pipeline code.
- Use `model_id` key from registry (not `hf_id` — that was the old BERT era).

---

## LLM serving

- Default: Groq `llama-3.3-70b-versatile` via `LLM_PROVIDER=groq`.
- HyDE: Gemini 2.0 Flash (`GEMINI_API_KEY`) → Groq fallback on rate limit or error.
- Swap via `LLM_PROVIDER` env var: `groq` | `lmstudio` | `anthropic`.
- LM Studio local option: `http://localhost:1234/v1`, model `gemma-4-e4b-it`.
- Never wire LLM calls directly in routes — always through `LLMBackend`.

---

## Key env vars

```
JWT_SECRET                    HS256 signing key
DATABASE_URL                  Supabase Postgres connection string
REDIS_URL                     Upstash Redis (rediss://)
CELERY_BROKER_URL             same as REDIS_URL
QDRANT_HOST                   Qdrant Cloud host
QDRANT_API_KEY                Qdrant Cloud API key
VOYAGE_API_KEY                Voyage AI (embeddings + reranker)
VOYAGE_EMBEDDING_MODEL        voyage-4 (default)
VOYAGE_RERANKER_MODEL         rerank-2.5
GROQ_API_KEY                  primary LLM + RAGAS eval
GROQ_API_KEY_2                secondary key for RAGAS rotation
GEMINI_API_KEY                HyDE rewriter
TAVILY_API_KEY                web search fallback
R2_ACCOUNT_ID                 Cloudflare R2
R2_ACCESS_KEY_ID
R2_SECRET_ACCESS_KEY
R2_BUCKET_NAME
GOOGLE_CLIENT_ID              Google OAuth
GOOGLE_CLIENT_SECRET
GOOGLE_REDIRECT_URI           {backend_url}/auth/google/callback
FRONTEND_URL                  Vercel URL (or localhost:3000 for dev)
LANGCHAIN_API_KEY             LangSmith tracing
```

---

## Custom skills (in `.claude/skills/`)

Invoke via `/{skill-name}` in Claude Code.

**Tenant ops**
- `/new-tenant` — scaffold a test tenant + JWT
- `/ingest-demo` — seed all demo tenants with their datasets
- `/teardown-tenant` — clean deletion across all stores

**Retrieval debugging**
- `/debug-retrieval` — full retrieval waterfall for a query
- `/trace-query` — LangGraph node-by-node execution trace
- `/inspect-chunks` — dump all chunks for a document with neighbors

**Evaluation + safety**
- `/check-isolation` — cross-tenant leak regression test
- `/eval-rag` — RAGAS scorecard on held-out questions
- `/benchmark-retrieval` — compare new config vs baseline
- `/replay-query` — re-run a LangSmith run against current code, show diff

**Environment + maintenance**
- `/reset-env` — wipe all state clean (with confirmation)
- `/healthcheck` — ping all services + row counts
- `/tail-logs` — stream structured logs filtered by tenant_id

**Documentation**
- `/update-checklist` — sync build order to reality from git log
- `/api-docs` — generate Markdown API reference from FastAPI routes

---

## Build order (checklist)

All backend items complete. Frontend deployed to Vercel.

- [x] docker-compose scaffold (local dev: Postgres + Redis + Qdrant)
- [x] Postgres schema (`tenants`, `workspaces`, `documents`, `eval_results`, `eval_strategies`)
- [x] JWT auth + `get_tenant()` dependency
- [x] Google OAuth 2.0 — `/auth/google` + `/auth/google/callback` + tenant auto-provisioning
- [x] Model registry (Voyage AI models, `model_id` key)
- [x] Voyage AI Embedder class
- [x] Workspace CRUD — POST/GET/DELETE `/workspaces` + Qdrant collection provisioning
- [x] Cloudflare R2 storage backend
- [x] Chunker + metadata extractor (SentenceSplitter default + tree-sitter AST for technical)
- [x] Celery ingestion task (Upstash broker, domain-specific queues)
- [x] POST `/ingest` — upload to R2, enqueue, return doc_id
- [x] HyDE rewriter (Gemini primary → Groq fallback)
- [x] Hybrid retrieval (SPLADE sparse + Voyage dense + RRF)
- [x] Voyage reranker + threshold gate (0.65)
- [x] LangGraph CRAG agent (retrieve → rerank → threshold → generate/web search)
- [x] Groq LLMBackend (default) + Anthropic optional
- [x] POST `/query` + SSE streaming `/query/stream`
- [x] Streamlit demo UI (legacy, superseded by Next.js frontend)
- [x] `test_isolation.py` passing
- [x] All 15 custom skills scaffolded in `.claude/skills/`
- [x] RAGAS eval harness (Groq key rotation + Voyage embeddings)
- [x] LangSmith tracing (automatic via `LANGCHAIN_TRACING_V2`)
- [x] Prometheus metrics + Grafana dashboard
- [x] Next.js 14 frontend — dashboard, chat (SSE), docs upload, eval, strategies, profile
- [x] Vercel deployment (semantiq-frontend-henna.vercel.app)
- [x] Dockerfile + .dockerignore for Cloud Run deployment
- [ ] GCP Cloud Run deployment (API + Celery worker)
- [ ] Update GOOGLE_REDIRECT_URI + FRONTEND_URL to production URLs
- [ ] Update Vercel NEXT_PUBLIC_API_URL to Cloud Run URL
- [ ] README Loom demo link (add after recording)
