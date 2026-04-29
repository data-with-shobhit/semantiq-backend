---
name: api-docs
description: Generate Markdown API reference by walking FastAPI routes. Produces request/response schemas, example curl commands with JWT header, and plan requirements per route. Writes to docs/API.md. Trigger on "api docs", "generate api docs", "document api", "api reference", "/api-docs".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# api-docs

Walks every FastAPI route under `api/routes/`, extracts Pydantic schemas, and
writes a complete Markdown API reference to `docs/API.md`.

## Invocation

```
/api-docs
```

## Playbook

### Step 1 — Discover all routes

```python
import importlib, inspect
from fastapi import FastAPI
from fastapi.routing import APIRoute

# Import the app to get registered routes
import sys
sys.path.insert(0, ".")
from api.main import app

routes = [r for r in app.routes if isinstance(r, APIRoute)]
routes.sort(key=lambda r: (r.path, list(r.methods)[0]))
```

Also scan `api/routes/*.py` directly to catch any routes that might not be
registered at app startup time.

Print: `Found <N> routes across <M> route files`

### Step 2 — Extract schema information

For each route, extract:
- HTTP method and path
- Summary / docstring
- Request body schema (Pydantic model fields + types + descriptions)
- Response schema (Pydantic model or plain dict)
- Dependencies (check for `Depends(get_tenant)` to identify authenticated routes)
- Plan requirements (check for plan guards in the handler body)

```python
for route in routes:
    handler = route.endpoint
    sig = inspect.signature(handler)

    # Detect authentication requirement
    requires_auth = any(
        "get_tenant" in str(p.default)
        for p in sig.parameters.values()
        if hasattr(p.default, "__name__") or hasattr(p.default, "dependency")
    )

    # Extract request body model
    body_model = route.body_field.type_ if route.body_field else None

    # Get response model
    response_model = route.response_model
```

### Step 3 — Build the Markdown document

Write `docs/API.md` with this structure:

```markdown
# RAG Platform API Reference

Generated: <timestamp>
Base URL: http://localhost:8000

All authenticated routes require:
```
Authorization: Bearer <JWT>
```

Get a JWT by signing up via `POST /signup` or running `/new-tenant` in Claude Code.

---

## Table of Contents
- [POST /signup](#post-signup)
- [POST /ingest](#post-ingest)
- [POST /query](#post-query)

---

## POST /signup

**Authentication**: Not required  
**Plan required**: None  
**Description**: Provision a new tenant. Creates Postgres row, empty Qdrant collection, storage directory, and returns a signed JWT.

### Request body

| Field  | Type   | Required | Description             |
|--------|--------|----------|-------------------------|
| name   | string | Yes      | Human-readable tenant name |

### Response (201 Created)

| Field      | Type   | Description                     |
|------------|--------|---------------------------------|
| tenant_id  | string | Unique tenant identifier        |
| token      | string | Signed JWT (valid 30 days)       |
| collection | string | Qdrant collection name          |

### Example

```bash
curl -X POST http://localhost:8000/signup \
  -H "Content-Type: application/json" \
  -d '{"name": "my-company"}'
```

### Response

```json
{
  "tenant_id": "my-company",
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "collection": "my-company_docs"
}
```

---

## POST /ingest

**Authentication**: Required (JWT)  
**Plan required**: free, pro, enterprise  
**Description**: Upload a document for ingestion. File is saved to storage, a Celery task is enqueued, and a job_id is returned immediately (202 Accepted).

### Request

Multipart form data:

| Field | Type | Required | Notes                          |
|-------|------|----------|--------------------------------|
| file  | file | Yes      | PDF, TXT, MD. Max 10 MB.       |

### Response (202 Accepted)

| Field  | Type   | Description                          |
|--------|--------|--------------------------------------|
| job_id | string | Celery task ID for status polling     |
| doc_id | int    | Postgres document row ID             |

### Example

```bash
export TOKEN="<your-jwt>"

curl -X POST http://localhost:8000/ingest \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@tesla-2023-10k.pdf"
```

---

## POST /query

**Authentication**: Required (JWT)  
**Plan required**: free, pro, enterprise  
**Description**: Ask a question against the tenant's document store. Runs the full retrieval pipeline: cache check → HyDE rewrite → BM25+dense hybrid → reranker → threshold gate → LLM generation.

### Request body

| Field    | Type   | Required | Default | Notes                    |
|----------|--------|----------|---------|--------------------------|
| question | string | Yes      | —       | Max 512 characters       |
| filters  | object | No       | {}      | Metadata pre-filters     |

### Response (200 OK)

| Field    | Type   | Description                                    |
|----------|--------|------------------------------------------------|
| answer   | string | Grounded answer with [Source N] citations      |
| chunks   | array  | Up to 5 cited chunks with scores               |
| trace_id | string | LangSmith trace ID for debugging               |
| cached   | bool   | True if result was served from semantic cache  |

### Example

```bash
export TOKEN="<your-jwt>"

curl -X POST http://localhost:8000/query \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What was revenue in 2023?"}'
```

---

## Error responses

| Status | Code                  | Meaning                                           |
|--------|-----------------------|---------------------------------------------------|
| 401    | unauthorized          | Missing or invalid JWT                            |
| 403    | plan_limit_exceeded   | Tenant over quota for their plan                  |
| 413    | file_too_large        | Upload exceeds 10 MB limit                        |
| 422    | validation_error      | Request body failed Pydantic validation           |
| 429    | rate_limit_exceeded   | Too many requests from this tenant                |
| 503    | retrieval_unavailable | Qdrant or embedding model temporarily unavailable |
```

### Step 4 — Write to file

```python
os.makedirs("docs", exist_ok=True)
with open("docs/API.md", "w") as f:
    f.write(api_doc_content)
```

Print: `API reference written to docs/API.md (<N> routes documented)`

### Step 5 — Validate examples

For each curl example in the generated docs, check that:
- The endpoint path actually exists in the app
- The request body fields match the Pydantic model
- The response fields match the response model

Print any discrepancies as warnings below the file path.
