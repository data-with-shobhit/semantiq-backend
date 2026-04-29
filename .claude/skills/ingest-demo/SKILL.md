---
name: ingest-demo
description: One-shot demo seeding. Wipes and recreates acme-corp and contoso tenants, ingests Tesla 10-K and FastAPI docs PDFs, waits for jobs, prints JWTs and chunk counts. Trigger on "seed demo", "ingest demo", "demo data", "/ingest-demo".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# ingest-demo

Reproducibly seeds the two canonical demo tenants used to prove tenant isolation.

## Invocation

```
/ingest-demo
```

No arguments. Idempotent — safe to run multiple times.

## Playbook

### Step 1 — Wipe existing demo tenants

For each of `acme-corp` and `contoso`, call the teardown sequence (same logic
as the `teardown-tenant` skill) without the confirmation prompt:

1. Delete Qdrant collection `{tenant_id}_docs` (ignore 404).
2. Flush Redis keys matching `{tenant_id}:*`.
3. Delete `storage/docs/{tenant_id}/` directory tree.
4. `DELETE FROM documents WHERE tenant_id = $1`
5. `DELETE FROM usage WHERE tenant_id = $1`
6. `DELETE FROM tenants WHERE id = $1`

Log each step: `{"event": "demo_teardown", "tenant_id": tenant_id}`

### Step 2 — Recreate tenants

Use `ingestion/model_registry.py` as the source of truth for embed_model and
embed_dim. Never hardcode model IDs or dims here.

```python
from ingestion.model_registry import MODEL_REGISTRY
fin = MODEL_REGISTRY["financial"]   # {"hf_id": "yiyanghkust/finbert-tone", "dim": 768}
tec = MODEL_REGISTRY["technical"]   # {"hf_id": "microsoft/codebert-base",   "dim": 768}
```

**acme-corp** (financial):
```sql
INSERT INTO tenants (id, plan, domain, embed_model, embed_dim, quota_docs, quota_tokens)
VALUES ('acme-corp', 'pro', 'financial', 'yiyanghkust/finbert-tone', 768, 500, 10000000);
```
Qdrant collection: `acme-corp_docs`, vector size 768 (FinBERT).

**contoso** (technical):
```sql
INSERT INTO tenants (id, plan, domain, embed_model, embed_dim, quota_docs, quota_tokens)
VALUES ('contoso', 'free', 'technical', 'microsoft/codebert-base', 768, 100, 1000000);
```
Qdrant collection: `contoso_docs`, vector size 768 (CodeBERT).
Note: NOT 384 — technical domain uses CodeBERT (768-dim), not MiniLM.

Create storage directories for both.

### Step 3 — Download PDFs if not cached

Cache directory: `data/`

**Tesla 10-K:**
- Cache path: `data/tesla-2023-10k.pdf`
- Source: SEC EDGAR full-submission filing for Tesla FY2023
- If not cached: `curl -L -o data/tesla-2023-10k.pdf "<sec-url>"`
- Verify file size > 1 MB before proceeding

**FastAPI docs:**
- Cache path: `data/fastapi-docs.pdf`
- If not cached: download from the official FastAPI docs PDF export URL
- Verify file size > 100 KB

Print: `Cache status: tesla-2023-10k.pdf (X MB) | fastapi-docs.pdf (Y MB)`

### Step 4 — Ingest both documents

For each document, call the ingestion pipeline directly (not via HTTP — use
the Celery task function directly for speed):

```python
from ingestion.tasks import process_document

# Insert document record first
doc_id = db.execute(
    "INSERT INTO documents (tenant_id, filename, path, status) VALUES ($1,$2,$3,'processing') RETURNING id",
    tenant_id, filename, file_path
)
# Run task synchronously for demo seeding (bypasses domain queue routing)
result = process_document.apply(args=[tenant_id, doc_id, file_path])
```

In production, tasks route to domain-specific Celery queues (`financial`,
`technical`, `general`, etc.) based on the tenant's `domain` field. For demo
seeding we call `.apply()` directly to stay synchronous.

Run both in parallel using `concurrent.futures.ThreadPoolExecutor`.

### Step 5 — Wait and poll

Poll Postgres every 2 seconds:
```sql
SELECT status, chunk_count FROM documents WHERE tenant_id = $1 ORDER BY id DESC LIMIT 1;
```

Print progress indicator: `acme-corp: processing... | contoso: ready (342 chunks)`

Timeout after 300 seconds — print error and partial status if exceeded.

### Step 6 — Issue JWTs and print summary

Issue 30-day JWTs for both tenants (same logic as `new-tenant` skill).

Print final summary:

```
=== Demo Seeding Complete ===

Tenant: acme-corp
  Domain  : financial
  Document: tesla-2023-10k.pdf
  Chunks  : 342
  JWT     : export ACME_TOKEN="<jwt>"

Tenant: contoso
  Domain  : technical
  Document: fastapi-docs.pdf
  Chunks  : 198
  JWT     : export CONTOSO_TOKEN="<jwt>"

Demo query (run this twice with different tokens to prove isolation):
  curl -H "Authorization: Bearer $ACME_TOKEN"  http://localhost:8000/query -d '{"question": "What was Tesla revenue in 2023?"}'
  curl -H "Authorization: Bearer $CONTOSO_TOKEN" http://localhost:8000/query -d '{"question": "What was Tesla revenue in 2023?"}'
```

## Error handling

- PDF download fails → print the manual download URL and expected cache path
- Qdrant collection creation fails on vector size mismatch → drop and recreate
- Ingestion task fails → print the last 20 lines of Celery worker logs
