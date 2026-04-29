# Architecture — Semantiq RAG Platform

Technical reference for the platform. Captures *why* each decision was made. Read before making architectural changes.

---

## 1. System overview

Semantiq is a multi-tenant CRAG (Corrective Retrieval-Augmented Generation) platform. Each tenant:

1. **Signs up** via Google OAuth → tenant space provisioned with a domain-specific Voyage AI embedding model
2. **Creates workspaces** → isolated Qdrant collection per workspace
3. **Uploads documents** → async pipeline: parse → chunk → Voyage embed → Qdrant + R2 + Supabase
4. **Asks questions** → CRAG: HyDE rewrite → hybrid retrieval → Voyage rerank → Groq LLM → SSE stream

### Full request flow

```
Browser (Vercel/Next.js)
        │
        ▼
FastAPI (Cloud Run)
        │
        ├─── JWT middleware → tenant_id from sub claim
        │
   ┌────┴──────────────────────┐
   │                           │
 /ingest                     /query/stream
   │                           │
   ▼                           ▼
Save to R2                 Check Upstash semantic cache
   │                           │ miss
   ▼                           ▼
Postgres doc row           LangGraph CRAG agent:
(status: processing)           │
   │                           ├─▶ Rewrite node
   ▼                           │   Gemini 2.0 Flash HyDE
Celery (Upstash broker)        │   (→ Groq fallback on 429)
   │                           │
   ▼                           ├─▶ Retrieve node
Parse (pypdf/unstructured)     │   SPLADE sparse + Voyage dense
   │                           │   RRF fusion (k=60)
   ▼                           │
Chunk (SentenceSplitter        ├─▶ Rerank node
       or tree-sitter AST)     │   Voyage rerank-2.5
   │                           │   threshold gate (< 0.65 → fallback)
   ▼                           │
Voyage AI embed                └─▶ Generate node
   │                               Groq llama-3.3-70b
   ▼                               SSE token stream
Qdrant Cloud upsert
Postgres status: ready
```

---

## 2. Multi-tenancy

### 2.1 Isolation strategy

**One Qdrant collection per workspace** (not per tenant — workspaces are the unit of isolation).

Collection name: `{tenant_id}_{workspace_id}_docs`

Why workspace-level (not tenant-level):
- Tenants may have multiple independent projects with different domains
- Domain selection (and therefore embedding model) is per-workspace
- Clean teardown — delete workspace = delete collection

| Layer | Isolation mechanism |
|---|---|
| Auth | JWT `sub` claim → `tenant_id`; extracted by middleware, never from request body |
| Postgres | `WHERE tenant_id = $1` on every query |
| Qdrant | Collection name computed server-side: `{tenant_id}_{ws_id}_docs` |
| Redis | All keys prefixed `{tenant_id}:` |
| R2 storage | Path prefix `docs/{tenant_id}/` |

`tenant_id` is **never** accepted from user input, URL params, or request body.

### 2.2 Provisioning flow (Google OAuth signup)

`GET /auth/google/callback` runs:

1. Exchange Google code → access token → fetch email
2. Derive `tenant_id` from email local part (slug-safe)
3. Upsert into Supabase `tenants` table (link `google_sub` on first login)
4. Auto-create a "Default" workspace with `general` domain
5. Create Qdrant collection `{tenant_id}_docs` with Voyage voyage-4-lite dimension (512)
6. Issue signed JWT (`sub = tenant_id`, 30-day expiry)
7. Redirect to `{FRONTEND_URL}/callback?token=<jwt>`

### 2.3 Workspace creation

`POST /workspaces` runs:

1. Look up embedding model from `model_registry[domain]`
2. Insert into Supabase `workspaces` table
3. Create Qdrant collection `{tenant_id}_{ws_id}_docs` with correct vector dim
4. Create sparse vector index for SPLADE
5. Create payload index on `section_num` for pre-filtering

---

## 3. Embedding strategy

### 3.1 Voyage AI — API-based embeddings

All embeddings are served by Voyage AI's REST API. No local model loading.

| Domain | Voyage Model | Dim | Strength |
|---|---|---|---|
| general (default) | `voyage-4-lite` | 512 | Fast, strong general retrieval |
| financial | `voyage-finance-2` | 1024 | Trained on financial corpora, SEC filings |
| legal | `voyage-law-2` | 1024 | Trained on legal documents, case law |
| medical | `voyage-4-lite` | 512 | General; medical domain uses clinical context |
| clinical | `voyage-4-lite` | 512 | General + clinical note context |
| scientific | `voyage-4-lite` | 512 | General + scientific paper context |
| technical | `voyage-code-3` | 1024 | Trained on code, APIs, documentation |

Why Voyage AI over local BERTs:
- **Quality** — Voyage models rank #1–2 on MTEB retrieval benchmarks
- **No GPU required** — API removes hardware dependency from deployment
- **Domain specialization** — `voyage-finance-2` and `voyage-law-2` are purpose-built, outperform general models by 8–15% on domain tasks
- **Consistent dimensions** — eliminates the 768/1024 mixed-dim complexity of local BERTs
- **Voyage code-3** handles polyglot code (Python, JS, TS, Go, Rust) better than CodeBERT

Model is locked at workspace creation. Mixing models within a collection breaks vector space consistency.

### 3.2 Model registry

Single source of truth: `ingestion/model_registry.py`

```python
MODEL_REGISTRY = {
    "general":    {"model_id": "voyage-4-lite",    "dim": 512},
    "financial":  {"model_id": "voyage-finance-2", "dim": 1024},
    "legal":      {"model_id": "voyage-law-2",     "dim": 1024},
    "medical":    {"model_id": "voyage-4-lite",    "dim": 512},
    "clinical":   {"model_id": "voyage-4-lite",    "dim": 512},
    "scientific": {"model_id": "voyage-4-lite",    "dim": 512},
    "technical":  {"model_id": "voyage-code-3",    "dim": 1024},
}
```

Never hardcode model IDs in pipeline code. Always read from registry.

---

## 4. Ingestion pipeline

### 4.1 Flow

```
POST /ingest?workspace_id=N (multipart file + JWT)
   │
   ├─▶ Validate file type + size
   ├─▶ Save to Cloudflare R2: docs/{tenant_id}/{filename}
   ├─▶ Insert Postgres row: documents (status=processing)
   ├─▶ Enqueue Celery task → Upstash Redis broker
   └─▶ Return 202 { doc_id, job_id }

Celery worker (async):
   ├─▶ Load file from R2
   ├─▶ Parse (pypdf / unstructured)
   ├─▶ Chunk (domain-aware)
   ├─▶ Extract metadata per chunk
   ├─▶ Voyage AI batch embed (chunks)
   ├─▶ SPLADE sparse encode
   ├─▶ Qdrant upsert (dense + sparse vectors)
   ├─▶ Postgres update: status=ready, chunk_count=N, strategy_id=S
   └─▶ Save chunking strategy to eval_strategies table
```

### 4.2 Parsing

Direct libraries — no LangChain loaders:

| File type | Parser |
|---|---|
| PDF | `pypdf.PdfReader` → fallback `unstructured.partition.pdf` for scanned/complex |
| DOCX | `python-docx` |
| TXT / MD | direct `file.read()` |
| PY / JS / TS | direct read → AST chunker |

### 4.3 Chunking

Domain-aware chunking strategy selected at ingestion time and saved to `eval_strategies` for analysis.

**Default (all non-technical domains):**

`llama-index-core` `SentenceSplitter` — respects sentence boundaries, paragraph breaks.

```python
SentenceSplitter(chunk_size=512, chunk_overlap=64, paragraph_separator="\n\n")
```

**Technical domain — AST-based:**

`tree-sitter` parses source code and splits at function/class definition boundaries. Each chunk = one complete syntactic unit. Preserves decorator context.

Fallback: `SentenceSplitter` if no top-level definitions found (config files, scripts).

**LLM-assisted strategy selection:**

The chunking strategy is analyzed by an LLM after ingestion. The LLM reviews the document structure, chunk quality, and domain, then selects or recommends strategy parameters. Results are stored in `eval_strategies` and visible in the Strategy Library UI.

### 4.4 Metadata per chunk

Every Qdrant payload includes:

| Field | Purpose |
|---|---|
| `doc_id` | Filter to specific document during eval |
| `tenant_id` | Safety double-check (collection already isolates) |
| `workspace_id` | Workspace scoping |
| `chunk_idx` | Chunk ordering |
| `section` | Structural filter (extracted by metadata extractor) |
| `section_num` | Numeric index on section (Qdrant payload index) |
| `has_numbers` | Route numeric queries |
| `page` | Citation precision |
| `text` | Raw chunk text (returned with results) |

---

## 5. Query pipeline (CRAG)

### 5.1 LangGraph CRAG agent

The agent is a LangGraph state graph with these nodes:

```
START → rewrite → retrieve → rerank → [threshold check] → generate → END
                                              │
                                         score < 0.65
                                              │
                                         web_search (Tavily) → generate → END
```

CRAG (Corrective RAG) means: if retrieval confidence is too low, fall back to web search before generating, rather than returning "I don't know."

### 5.2 Query rewriting — HyDE

**HyDE (Hypothetical Document Embeddings)**:
- The LLM generates a hypothetical answer to the question
- That answer is embedded and used as the search vector
- Aligns query vector with answer-space chunks (not question-space)

Implementation:
- Primary: Gemini 2.0 Flash (fast, cheap)
- Fallback: Groq llama-3.3-70b (on any exception including 429 rate limit)
- Final fallback: original query (pass-through)

### 5.3 Hybrid retrieval

Two retrievers run in parallel, fused with Reciprocal Rank Fusion (RRF):

**Sparse — SPLADE primary, BM25 fallback:**

- `naver/splade-v3` learned sparse model with implicit vocabulary expansion
- Stored as Qdrant named sparse vectors, queried with `using="sparse"`
- Falls back to `rank-bm25` on empty result or exception

**Dense — Voyage AI:**

- Tenant's domain-specific Voyage model
- Queried with `using="dense"` against workspace collection

**RRF fusion:**

```python
def rrf_fuse(sparse_hits, dense_hits, k=60):
    scores = defaultdict(float)
    for rank, hit in enumerate(sparse_hits):
        scores[hit.id] += 1 / (k + rank)
    for rank, hit in enumerate(dense_hits):
        scores[hit.id] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])
```

`k=60` is rank-based — no score normalization needed between SPLADE and cosine scores.

### 5.4 Reranking

Top-20 fused candidates → Voyage `rerank-2.5` → top-5 reranked chunks.

CrossEncoder-style: sees query + candidate together, captures fine-grained relevance that bi-encoder cosine similarity misses.

### 5.5 Threshold gate

- Score ≥ 0.65 → proceed to generate
- Score < 0.65 → web search via Tavily → generate with web results
- No result from web → return "I don't have that information in your documents"

### 5.6 Generation

Groq `llama-3.3-70b-versatile` via OpenAI-compatible client.

System prompt rules (strictly enforced):
- Every claim must be in sources — no inference from training knowledge
- No source labels in answer (no "According to DOC Source 1")
- If sources don't cover the question → exact phrase "I don't have that information in your documents"
- Answers in plain natural language

Conversation history: last 4 turns trimmed before sending.

### 5.7 SSE streaming

`/query/stream` returns Server-Sent Events:

```
data: {"type": "status", "node": "retrieve"}
data: {"type": "status", "node": "rerank"}
data: {"type": "token", "data": "The "}
data: {"type": "token", "data": "answer "}
data: {"type": "done", "llm_calls": 2, "chunks": [...]}
```

Frontend displays node status with a `⚡ RETRIEVING` indicator while streaming, then renders the full answer with source chunks and latency meta.

### 5.8 Semantic cache

Before running the CRAG agent, check Upstash Redis for a cached result:
- Key: `{tenant_id}:{workspace_id}:query:{semantic_hash}`
- TTL: 1 hour
- `bypass_cache: true` skips cache (used during eval)

---

## 6. LLM serving

### 6.1 Current stack

| Role | Model | Provider |
|---|---|---|
| Generation | `llama-3.3-70b-versatile` | Groq |
| HyDE rewrite | `gemini-2.0-flash` | Google AI Studio |
| HyDE fallback | `llama-3.3-70b-versatile` | Groq |
| RAGAS eval | Groq (key rotation across 2 keys) | Groq |
| Optional | `google/gemma-4-31b-it:free` | OpenRouter |

### 6.2 LLMBackend abstraction

```python
class LLMBackend(Protocol):
    async def generate(self, prompt: str, system: str = "") -> str: ...
    async def stream(self, prompt: str, system: str = "") -> AsyncIterator[str]: ...
```

Implementations: `GroqBackend` (default), `AnthropicBackend` (optional), `OpenRouterBackend`.

Swap via `LLM_PROVIDER` env var. No route code changes needed.

### 6.3 Why Groq over local Ollama

- **Latency** — Groq's GroqChip delivers ~500 tokens/sec vs ~30 tok/sec on local GPU
- **No GPU required** — removes hardware dependency from Cloud Run deployment
- **Context** — llama-3.3-70b supports 128K context, sufficient for 5 chunks + history
- **Cost** — $0.59/$0.79 per million tokens (input/output) — negligible at demo scale

Local Ollama/LM Studio remains an option via `LLM_PROVIDER=lmstudio` for offline use.

---

## 7. Storage layer

### 7.1 Cloudflare R2

All uploaded files stored in R2 bucket (`ragproject2002`).

Path: `docs/{tenant_id}/{filename}`

R2 chosen over S3 for zero egress fees — the backend reads files during ingestion, which would incur S3 data transfer costs.

```python
class StorageBackend(Protocol):
    def save(self, tenant_id: str, filename: str, data: bytes) -> str: ...
    def load(self, tenant_id: str, filename: str) -> bytes: ...
    def delete(self, tenant_id: str, filename: str) -> None: ...
```

`STORAGE_BACKEND=r2` in production. `local` for dev without credentials.

### 7.2 Supabase (Postgres)

Managed Postgres on AWS ap-southeast-1. Connection via `asyncpg` with connection pooling via Supabase's PgBouncer.

Key tables:

```sql
tenants        -- id (slug), plan, domain, embed_model, embed_dim, google_sub, email
workspaces     -- id, tenant_id, name, domain, embed_model, embed_dim, collection_name
documents      -- id, workspace_id, tenant_id, filename, status, chunk_count, strategy_id
eval_results   -- id, doc_id, tenant_id, faithfulness, context_precision, answer_relevance
eval_strategies -- id, doc_id, chunker, chunk_size, overlap, reasoning, is_active
```

### 7.3 Upstash Redis

Serverless Redis (TLS) for:

| Use case | Key pattern | TTL |
|---|---|---|
| Semantic query cache | `{tenant_id}:{ws_id}:query:{hash}` | 1 hour |
| Celery task broker | Upstash native | — |
| Celery result backend | Upstash native | 24 hours |

---

## 8. Frontend architecture

### 8.1 Stack

Next.js 14 App Router, TypeScript, Tailwind CSS, deployed on Vercel.

| Library | Role |
|---|---|
| Zustand | Auth token (persisted to localStorage) |
| TanStack Query | Server state, auto-refetch, optimistic updates |
| React Dropzone | Drag-and-drop file upload |
| Sonner | Toast notifications |
| Lucide React | Icons |

### 8.2 Route structure

```
/                     Landing page (aurora background)
/signin               Google OAuth entry point
/callback             OAuth token receiver → stores JWT → /dashboard
/dashboard            Workspace grid + storage usage
/profile              Tenant info + sign out
/workspace/[id]/chat      SSE streaming chat with source chunks
/workspace/[id]/docs      Upload + document list with ingestion status
/workspace/[id]/eval      RAGAS evaluation runner + score bars
/workspace/[id]/strategies  Strategy library with name/save
```

Route groups `(auth)` and `(app)` are Next.js grouping only — don't appear in URLs.

### 8.3 Auth flow

```
/signin → window.location = /auth/google (FastAPI)
        → Google consent
        → FastAPI /auth/google/callback
        → JWT issued
        → redirect /callback?token=<jwt>
        → Zustand setToken → /dashboard
```

JWT stored in Zustand with localStorage persistence. Cleared on 401 or explicit sign out.

### 8.4 Chat history

Per-workspace, stored in localStorage:
- Key: `semantiq:chat:{workspaceId}`
- TTL: 12 hours from last message
- Cleared on "New Chat" button or TTL expiry

---

## 9. Observability

### 9.1 LangSmith tracing

Every `/query` call traces each LangGraph node automatically via `LANGCHAIN_TRACING_V2=true`. Visible at `smith.langchain.com` — query rewrite, retrieved chunks + scores, reranker scores, LLM prompt.

### 9.2 Prometheus metrics

Exposed at `/metrics`. Scraped by Prometheus. Dashboard in `monitoring/grafana_dashboard.json`.

Key metrics: `rag_queries_total`, `rag_query_latency_seconds`, `rag_cache_hits_total`, `rag_ingest_total`, `rag_rerank_threshold_fail_total`.

### 9.3 RAGAS evaluation

Per-document evaluation:
- Questions + ground truths entered in UI
- Each Q runs through full CRAG pipeline with `bypass_cache=true`
- Faithfulness scored by Groq LLM
- Context precision scored by Groq LLM
- Answer relevancy scored by Groq + Voyage embeddings
- 5s delay between questions to stay under Groq RPM limits
- Results saved to Supabase `eval_results` table

---

## 10. Deployment architecture

```
GitHub (semantiq-frontend-)     GitHub (semantiq-backend)
        │                               │
        ▼                               ▼
    Vercel                      GCP Artifact Registry
  (auto-deploy)                         │
        │                        ┌──────┴──────┐
        ▼                        ▼             ▼
semantiq-frontend-          Cloud Run      Cloud Run
  henna.vercel.app           (API)          (Worker)
                              │               │
                         Supabase        Upstash Redis
                         Qdrant Cloud    Cloudflare R2
```

Backend Cloud Run services share the same Docker image, different startup commands:
- API: `uvicorn api.main:app --host 0.0.0.0 --port 8080`
- Worker: `celery -A ingestion.tasks worker ...`

All credentials injected as Cloud Run environment variables — never in the image.

---

## 11. Security posture

- JWTs signed with HS256; secret from env, never hardcoded
- `tenant_id` extracted from JWT only — zero user-input trust
- Rate limiting per-tenant via `slowapi`
- File uploads: type allowlist + size cap (50MB)
- CORS: explicit allowlist (Vercel URL + localhost)
- R2 bucket: private, accessed via service credentials
- Supabase: SSL-enforced connections, row-level isolation via `WHERE tenant_id = $1`
- `tests/test_isolation.py` runs on every commit — cross-tenant data leak = build failure

---

## 12. What this architecture explicitly doesn't do

- Does not support cross-tenant queries or shared document pools
- Does not fine-tune embedding models at runtime
- Does not do OCR (scanned PDFs are out of scope — would need Tesseract)
- Does not mix embedding models within a collection — ever
- Does not accept `tenant_id` from user input — ever
- Does not stream Celery task progress in real time (polling via TanStack Query refetchInterval)
