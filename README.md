# Semantiq — Multi-Tenant RAG Platform

Production-grade, multi-tenant Retrieval-Augmented Generation (CRAG) platform. Each tenant gets logically isolated document stores, domain-specific embeddings via Voyage AI, and answers strictly grounded in their own data. Think Google Drive for RAG — one platform, completely separate data per tenant.

**Live frontend:** [semantiq-frontend-henna.vercel.app](https://semantiq-frontend-henna.vercel.app)

---

## What it does

1. **Sign up** — Google OAuth provisions your tenant space with a domain-specific Voyage AI embedding model
2. **Create workspaces** — organize documents by project or domain
3. **Upload documents** — PDF, DOCX, TXT, MD, PY, JS, TS. Celery worker parses, chunks, embeds async
4. **Ask questions** — CRAG pipeline: HyDE rewrite → hybrid retrieval (SPLADE sparse + Voyage dense + RRF) → Voyage reranker → Groq LLM → SSE-streamed answer
5. **Evaluate** — RAGAS scorecard (faithfulness, context precision, answer relevancy) per document
6. **Strategy library** — inspect and name chunking strategies used per document

---

## Architecture overview

```
Browser (Next.js / Vercel)
        │
        ▼
FastAPI (Cloud Run)  ──▶  JWT auth (Google OAuth)
        │
   ┌────┴────────────┐
   │                 │
Ingestion          Query
(Celery/Upstash)   (LangGraph CRAG)
   │                 │
   ▼                 ▼
Parse → Chunk    HyDE rewrite (Gemini → Groq fallback)
   │                 │
   ▼                 ▼
Voyage AI embed  Hybrid retrieve
   │             (SPLADE sparse + Voyage dense + RRF)
   ▼                 │
Qdrant Cloud     Voyage reranker
Cloudflare R2        │
Supabase         Threshold gate (score < 0.65 → "I don't know")
                     │
                 Groq LLM (llama-3.3-70b) → SSE stream
```

---

## Tech stack

| Layer | Technology |
|---|---|
| **Frontend** | Next.js 14 (App Router), Tailwind CSS, Zustand, TanStack Query |
| **API** | FastAPI, PyJWT, slowapi |
| **Auth** | Google OAuth 2.0 → JWT (tenant_id from `sub` claim) |
| **LLM** | Groq `llama-3.3-70b-versatile` (primary), OpenRouter Gemma 4 (fallback) |
| **HyDE** | Gemini 2.0 Flash (primary) → Groq fallback on rate limit |
| **Embeddings** | Voyage AI — `voyage-4-lite` (general/scientific/medical/clinical), `voyage-finance-2` (financial), `voyage-law-2` (legal), `voyage-code-3` (technical) |
| **Reranker** | Voyage AI `rerank-2.5` |
| **Sparse retrieval** | SPLADE `naver/splade-v3` + BM25 fallback |
| **Orchestration** | LangGraph ReAct (CRAG agent) |
| **Vector DB** | Qdrant Cloud (one collection per tenant) |
| **Postgres** | Supabase (tenants, workspaces, documents, eval results) |
| **Redis / Queue** | Upstash Redis (semantic cache + Celery broker) |
| **File storage** | Cloudflare R2 |
| **Eval** | RAGAS via Groq LLM + Voyage embeddings |
| **Tracing** | LangSmith |
| **Observability** | Prometheus + Grafana, structlog |
| **Deploy** | Vercel (frontend), GCP Cloud Run (backend + worker) |

---

## Multi-tenancy

`tenant_id` is always extracted server-side from the JWT `sub` claim. Never accepted from user input.

| Layer | Isolation |
|---|---|
| Qdrant | Collection `{tenant_id}_{workspace_id}_docs` per workspace |
| Postgres | `WHERE tenant_id = $1` on every query |
| Redis | Keys prefixed `{tenant_id}:` |
| R2 | Path prefix `docs/{tenant_id}/` |

Cross-tenant leak = build failure (`tests/test_isolation.py`).

---

## Embedding strategy

Model selected at workspace creation, locked for the collection lifetime. Mixing models in one collection breaks vector space consistency.

| Domain | Voyage Model | Dim |
|---|---|---|
| general (default) | `voyage-4-lite` | 512 |
| financial | `voyage-finance-2` | 1024 |
| legal | `voyage-law-2` | 1024 |
| medical | `voyage-4-lite` | 512 |
| clinical | `voyage-4-lite` | 512 |
| scientific | `voyage-4-lite` | 512 |
| technical | `voyage-code-3` | 1024 |

---

## Local development

### Prerequisites

- Python 3.11+, [uv](https://docs.astral.sh/uv/)
- Node.js 18+
- Docker Desktop (for Qdrant + Postgres + Redis locally, or use cloud services)

### 1. Clone and install

```bash
git clone https://github.com/data-with-shobhit/semantiq-backend.git
cd semantiq-backend
uv sync
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in all API keys — see .env.example for required vars
```

Key vars:
```env
VOYAGE_API_KEY=...
GROQ_API_KEY=...
GEMINI_API_KEY=...
DATABASE_URL=postgresql://...      # Supabase connection string
REDIS_URL=rediss://...             # Upstash Redis URL
QDRANT_HOST=...                    # Qdrant Cloud host
QDRANT_API_KEY=...
R2_ACCESS_KEY_ID=...               # Cloudflare R2
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=...
GOOGLE_CLIENT_ID=...               # Google OAuth
GOOGLE_CLIENT_SECRET=...
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
FRONTEND_URL=http://localhost:3000
JWT_SECRET=...
```

### 3. Run API

```bash
uv run uvicorn api.main:app --reload --port 8000
```

### 4. Run Celery worker

```bash
uv run celery -A ingestion.tasks worker \
  -Q general,financial,medical,clinical,legal,scientific,technical \
  --loglevel=info
```

### 5. Run frontend

```bash
cd ../semantiq-frontend
npm install
cp .env.local.example .env.local
# Set NEXT_PUBLIC_API_URL=http://localhost:8000
npm run dev   # http://localhost:3000
```

---

## API reference

All routes require `Authorization: Bearer <jwt>` except `/auth/google` and `/health`.

### Auth

```
GET  /auth/google             → redirect to Google consent screen
GET  /auth/google/callback    → exchange code, issue JWT, redirect to frontend
GET  /auth/me                 → current tenant profile
```

### Workspaces

```
GET    /workspaces            → list workspaces for tenant
POST   /workspaces            → create workspace { name, domain }
DELETE /workspaces/{id}       → delete workspace + Qdrant collection
```

### Documents

```
POST   /ingest?workspace_id=  → upload file, enqueue ingestion, return doc_id
GET    /ingest/list?workspace_id= → list documents with status
DELETE /ingest/{doc_id}       → delete document + Qdrant chunks
```

### Query

```
POST /query                   → { question, workspace_id, history, bypass_cache }
POST /query/stream            → SSE stream of { type: status|token|done, data }
```

### Eval

```
POST /eval/run                → { doc_id, questions[], ground_truths[], delay_s }
GET  /eval/history/{doc_id}   → latest RAGAS result for a document
GET  /eval/strategies         → list all chunking strategies
GET  /eval/strategy/{doc_id}  → strategy history for a document
POST /eval/strategy/{id}/name → name a strategy { name }
```

---

## RAGAS evaluation

From the frontend: Eval tab → select document → add Q&A pairs → Run Evaluation.

Metrics:
- **Faithfulness** — every claim in the answer must be in the retrieved chunks
- **Context Precision** — are the retrieved chunks actually relevant?
- **Answer Relevancy** — does the answer address the question?

Groq LLM scores each metric. Voyage embeddings used for answer relevancy cosine similarity. 5-second inter-question delay to stay under Groq RPM limits.

---

## Project structure

```
api/
├── main.py              FastAPI app + CORS + lifespan
├── auth.py              JWT middleware, get_tenant() dependency
└── routes/
    ├── auth.py          Google OAuth flow
    ├── workspaces.py    Workspace CRUD + Qdrant provisioning
    ├── ingest.py        File upload + Celery enqueue
    ├── query.py         RAG query + SSE streaming
    └── eval.py          RAGAS eval + strategy management

ingestion/
├── model_registry.py   domain → Voyage model mapping
├── embedder.py         Voyage AI embedding client
├── chunker.py          SentenceSplitter (default) + tree-sitter AST (technical)
├── loader.py           pypdf + unstructured wrappers
├── extractor.py        metadata extraction
└── tasks.py            Celery ingestion task

retrieval/
├── rewriter.py         HyDE (Gemini primary, Groq fallback)
├── splade.py           SPLADE sparse retrieval + BM25 fallback
├── hybrid.py           SPLADE + dense + RRF fusion
└── reranker.py         Voyage reranker

graph/
└── query_graph.py      LangGraph CRAG agent

llm/
├── backend.py          LLMBackend protocol
├── groq_backend.py     Groq (default)
└── anthropic_backend.py  optional Claude

storage/
├── backend.py          StorageBackend protocol
└── r2.py               Cloudflare R2 implementation

cache/
└── redis_client.py     Semantic cache + tenant-prefixed keys

db/
├── postgres.py         asyncpg connection + query helpers
└── schema.sql          tenants, workspaces, documents, eval_results

eval/
└── ragas_eval.py       RAGAS harness with Groq key rotation

tests/
├── test_isolation.py   CRITICAL: cross-tenant leak test
├── test_auth.py
└── test_retrieval.py

monitoring/
├── prometheus.yml
└── grafana_dashboard.json

.claude/skills/         15 dev workflow skills
```

---

## Deployment

### Frontend (Vercel)

Auto-deploys from `semantiq-frontend-` GitHub repo. Set env var:
- `NEXT_PUBLIC_API_URL` = Cloud Run backend URL

### Backend (GCP Cloud Run)

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/semantiq-backend
gcloud run deploy semantiq-backend \
  --image gcr.io/PROJECT_ID/semantiq-backend \
  --platform managed --region us-central1 \
  --allow-unauthenticated --port 8080 --memory 2Gi
```

Worker (Celery):
```bash
gcloud run deploy semantiq-worker \
  --image gcr.io/PROJECT_ID/semantiq-backend \
  --min-instances 1 \
  --command "celery" \
  --args "-A,ingestion.tasks,worker,--loglevel=info"
```

After deploy:
1. Update `GOOGLE_REDIRECT_URI` → Cloud Run URL
2. Update `FRONTEND_URL` → Vercel URL
3. Update `NEXT_PUBLIC_API_URL` in Vercel → Cloud Run URL
4. Add Cloud Run URL to Google OAuth authorized redirect URIs

---

## Tests

```bash
uv run pytest tests/ -v

# Critical — cross-tenant isolation must never fail
uv run pytest tests/test_isolation.py -v
```

---

## Custom Claude Code skills

| Skill | Purpose |
|---|---|
| `/new-tenant` | Scaffold tenant + JWT |
| `/debug-retrieval` | Full retrieval waterfall for a query |
| `/trace-query` | LangGraph node-by-node trace |
| `/inspect-chunks` | Dump chunks with neighbors |
| `/check-isolation` | Cross-tenant leak regression test |
| `/eval-rag` | RAGAS scorecard |
| `/healthcheck` | Ping all services |
| `/tail-logs` | Stream logs by tenant_id |
