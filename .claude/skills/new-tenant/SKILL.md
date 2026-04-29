---
name: new-tenant
description: Scaffold a test tenant. Creates Postgres row, empty Qdrant collection, storage directory, and issues a signed JWT. Trigger on "new tenant", "create tenant", "add tenant", "/new-tenant".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# new-tenant

Scaffold a new test tenant end-to-end from a single name argument.

## Invocation

```
/new-tenant <name>
```

`name` is a short slug, e.g. `acme-corp`. The tenant_id is derived from it.

## Playbook

### Step 1 — Resolve tenant_id

```python
import re, uuid
tenant_id = re.sub(r"[^a-z0-9-]", "-", name.lower()).strip("-")
# If blank after sanitization, fall back to a UUID4 slug
if not tenant_id:
    tenant_id = str(uuid.uuid4())[:8]
```

Print: `tenant_id = <value>`

### Step 2 — Resolve embed model and dimension from model_registry

```python
from ingestion.model_registry import MODEL_REGISTRY

# Accept domain as optional second arg; default to 'general'
domain = args.domain if args.domain else "general"
model_info = MODEL_REGISTRY[domain]  # raises KeyError if unknown domain
embed_model = model_info["hf_id"]
embed_dim   = model_info["dim"]
```

Valid domains and their dims (source of truth: `ingestion/model_registry.py`):

| Domain | Model | Dim |
|---|---|---|
| general (default) | BAAI/bge-m3 | 1024 |
| financial | yiyanghkust/finbert-tone | 768 |
| medical | dmis-lab/biobert-v1.1 | 768 |
| clinical | emilyalsentzer/Bio_ClinicalBERT | 768 |
| legal | nlpaueb/legal-bert-base-uncased | 768 |
| scientific | allenai/scibert_scivocab_uncased | 768 |
| technical | microsoft/codebert-base | 768 |

If domain is unknown, print the valid list and abort.

### Step 3 — Insert Postgres row

Connect via `db/postgres.py`. Run:

```sql
INSERT INTO tenants (id, plan, domain, embed_model, embed_dim, quota_docs, quota_tokens)
VALUES ($1, 'free', $2, $3, $4, 100, 1000000)
ON CONFLICT (id) DO NOTHING;
```

Parameters: `tenant_id, domain, embed_model, embed_dim`

Log with structlog: `{"event": "tenant_created", "tenant_id": tenant_id, "domain": domain, "embed_dim": embed_dim}`

If conflict: print a warning that the tenant already exists and ask whether to
continue or abort.

### Step 4 — Create Qdrant collection

Collection name: `{tenant_id}_docs`. Vector size comes from `embed_dim` resolved
in Step 2 — never hardcode.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)
client.recreate_collection(
    collection_name=f"{tenant_id}_docs",
    vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
)
```

Log: `{"event": "qdrant_collection_created", "tenant_id": tenant_id, "collection": f"{tenant_id}_docs", "embed_dim": embed_dim}`

### Step 5 — Create storage directory

```python
import os
path = f"storage/docs/{tenant_id}"
os.makedirs(path, exist_ok=True)
```

Print: `Storage directory: storage/docs/{tenant_id}/`

### Step 6 — Issue signed JWT

```python
import jwt, os, time
secret = os.environ.get("JWT_SECRET", "dev-secret-change-me")
token = jwt.encode(
    {"sub": tenant_id, "iat": int(time.time()), "exp": int(time.time()) + 86400 * 30},
    secret,
    algorithm="HS256",
)
```

Print the JWT clearly so it can be pasted into curl commands:

```
JWT (valid 30 days):
  export TOKEN="<jwt>"

Test with:
  curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/query \
       -d '{"question": "hello"}'
```

### Step 7 — Summary

Print a one-line summary table:

```
tenant_id  : <value>
plan       : free
domain     : <domain>
embed_model: <embed_model>
embed_dim  : <embed_dim>
collection : {tenant_id}_docs  (vector size: <embed_dim>)
storage    : storage/docs/{tenant_id}/
JWT        : <first 40 chars>...
```

## Error handling

- Postgres unreachable → print connection string, suggest `docker-compose up -d postgres`
- Qdrant unreachable → suggest `docker-compose up -d qdrant`
- JWT_SECRET not set → warn that dev-secret is in use, never use in production
