---
name: healthcheck
description: Ping all infrastructure services and print a status table. Checks Qdrant collections with vector counts, Postgres row counts, Redis memory, and local storage size. Trigger on "healthcheck", "health check", "service status", "check services", "/healthcheck".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# healthcheck

Pings every infrastructure layer and prints a color-coded status table.
Use this to quickly confirm the environment is up before running demos or tests.

## Invocation

```
/healthcheck
```

## Playbook

### Step 1 — Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
import time

start = time.perf_counter()
try:
    client = QdrantClient(host="localhost", port=6333)
    collections = client.get_collections().collections
    qdrant_ok = True
    qdrant_latency = int((time.perf_counter() - start) * 1000)

    qdrant_detail = []
    for c in collections:
        info = client.get_collection(c.name)
        qdrant_detail.append({
            "collection": c.name,
            "vectors": info.vectors_count,
            "indexed": info.indexed_vectors_count,
            "status": info.status,
        })
except Exception as e:
    qdrant_ok = False
    qdrant_error = str(e)
```

### Step 2 — Postgres

```python
start = time.perf_counter()
try:
    counts = db.fetch_one("""
        SELECT
          (SELECT COUNT(*) FROM tenants)   AS tenants,
          (SELECT COUNT(*) FROM documents) AS documents,
          (SELECT COUNT(*) FROM usage)     AS usage,
          (SELECT COUNT(*) FROM documents WHERE status = 'processing') AS processing,
          (SELECT COUNT(*) FROM documents WHERE status = 'failed') AS failed
    """)
    pg_ok = True
    pg_latency = int((time.perf_counter() - start) * 1000)
except Exception as e:
    pg_ok = False
    pg_error = str(e)
```

### Step 3 — Redis

```python
start = time.perf_counter()
try:
    info = redis_client.info()
    key_count = redis_client.dbsize()
    redis_ok = True
    redis_latency = int((time.perf_counter() - start) * 1000)
    redis_memory = info["used_memory_human"]
    redis_peak = info["used_memory_peak_human"]
    redis_version = info["redis_version"]
except Exception as e:
    redis_ok = False
    redis_error = str(e)
```

### Step 4 — Local storage

```python
import os

storage_root = "storage/docs"
try:
    total_bytes = 0
    tenant_dirs = []
    if os.path.exists(storage_root):
        for entry in os.scandir(storage_root):
            if entry.is_dir():
                size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, files in os.walk(entry.path)
                    for f in files
                )
                file_count = sum(len(files) for _, _, files in os.walk(entry.path))
                tenant_dirs.append((entry.name, file_count, size))
                total_bytes += size
    storage_ok = True
except Exception as e:
    storage_ok = False
    storage_error = str(e)
```

### Step 5 — Print status table

Use green/red indicators:
- `[OK]` for healthy services (use ANSI green if terminal supports it)
- `[DOWN]` for unreachable services (use ANSI red)
- `[WARN]` for degraded state (e.g., documents in `processing` > 5 minutes)

```
=== HEALTHCHECK ===
Timestamp: <ISO>

┌──────────────┬────────┬──────────────────────────────────────────────┐
│ Service      │ Status │ Details                                      │
├──────────────┼────────┼──────────────────────────────────────────────┤
│ Qdrant       │ [OK]   │ 3 collections, latency 12ms                  │
│ Postgres     │ [OK]   │ 3 tenants / 4 docs / 47 usage, latency 5ms   │
│ Redis        │ [OK]   │ 128 keys, 2.4MB / peak 3.1MB, latency 2ms    │
│ Storage      │ [OK]   │ 3 tenant dirs, 12 files, 45.2MB total        │
└──────────────┴────────┴──────────────────────────────────────────────┘

Qdrant collections:
  acme-corp_docs      342 vectors  (342 indexed)  green
  contoso_docs     198 vectors  (198 indexed)  green

Postgres details:
  Tenants   : 3
  Documents : 4 (0 processing, 0 failed)
  Usage rows: 47

Storage breakdown:
  acme-corp/      8 files    32.1MB
  contoso/     4 files    13.1MB
```

### Step 6 — Warn on anomalies

Print warnings below the table for:

- Documents stuck in `processing` for >5 minutes:
  ```
  WARN: 2 documents in 'processing' state for >5 minutes — Celery worker may be down
  ```

- Qdrant collection exists but has 0 vectors:
  ```
  WARN: Collection 'xyz_docs' has 0 vectors — ingestion may not have run
  ```

- Redis memory > 80% of `maxmemory`:
  ```
  WARN: Redis at 85% of maxmemory — consider flushing old cache entries
  ```

- Storage directory exists in filesystem but has no Postgres tenant row:
  ```
  WARN: Orphaned storage directory 'storage/docs/old-tenant' has no tenant row
  ```

### Step 7 — Overall summary

```
Overall: [OK] all services healthy
   OR
Overall: [DEGRADED] 1 warning(s)
   OR
Overall: [DOWN] Qdrant unreachable — run: docker-compose up -d qdrant
```
