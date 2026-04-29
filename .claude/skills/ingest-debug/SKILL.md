---
name: ingest-debug
description: >
  Debug the ingestion pipeline for a specific tenant and document. Traces the full
  path from file upload → Celery task → chunking → embedding → Qdrant upsert.
  Use when user says "debug ingestion", "ingest failing", "document not showing up",
  "why isn't my doc indexed", or invokes /ingest-debug.
---

## Diagnostic steps

Run these in order, stop at the first failure and report it:

### 1. File storage
- Check `storage/docs/{tenant_id}/` — is the file present?
- Check file size > 0 and extension is supported (.pdf, .txt, .md)

### 2. Celery task queue
- Check Redis for the task: `redis-cli KEYS "{tenant_id}:task:*"`
- Check Celery worker logs for the task_id
- Verify task status: PENDING → STARTED → SUCCESS (or FAILURE + traceback)

### 3. Chunking
- Run chunker manually on the file — does it produce chunks?
- Check chunk count > 0
- Verify metadata fields: `tenant_id`, `doc_id`, `chunk_index`, `fiscal_year`, `section`

### 4. Embedding
- Verify the right model is selected (FinBERT for financial tenants, general otherwise)
- Check embedding dimension matches Qdrant collection config
- Test: embed one chunk and print shape

### 5. Qdrant upsert
- Collection name: `{tenant_id}_docs`
- Check point count before and after upsert
- Verify payload includes `tenant_id` field on every point

### 6. Postgres record
- Check `documents` table: `SELECT * FROM documents WHERE tenant_id = '{tenant_id}' ORDER BY created_at DESC LIMIT 5`
- Verify status column = 'indexed'

## Output format

```
Step N — <name>: PASS | FAIL | SKIP
  Detail: <what was found>
  Fix: <if FAIL, exact command or code to fix>
```

Root cause summary at the end: one sentence.
