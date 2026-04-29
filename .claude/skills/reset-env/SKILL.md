---
name: reset-env
description: Full environment wipe. Drops all Qdrant collections, truncates all Postgres tables, flushes Redis, deletes storage/docs. Requires typing a confirmation phrase. Trigger on "reset env", "wipe everything", "clean slate", "full reset", "/reset-env".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# reset-env

Nuclear option — wipes all data from all layers. Use before a clean demo or
after a failed migration left the environment in a bad state.

## Invocation

```
/reset-env
```

## Playbook

### Step 1 — Audit what will be deleted

Before asking for confirmation, show exactly what will be wiped:

```python
# Qdrant
collections = client.get_collections().collections
qdrant_summary = [(c.name, client.get_collection(c.name).vectors_count) for c in collections]

# Postgres
pg_counts = db.fetch_one("""
SELECT
  (SELECT COUNT(*) FROM tenants)   AS tenants,
  (SELECT COUNT(*) FROM documents) AS documents,
  (SELECT COUNT(*) FROM usage)     AS usage
""")

# Redis
redis_key_count = len(redis_client.keys("*"))
redis_memory = redis_client.info("memory")["used_memory_human"]

# Storage
import os
total_files = sum(len(f) for _, _, f in os.walk("storage/docs"))
```

Print:
```
=== RESET PREVIEW ===

Qdrant collections to drop:
  acme-corp_docs           (342 vectors)
  contoso_docs          (198 vectors)
  isolation-test-abc123_docs (50 vectors)

Postgres rows to truncate:
  tenants:   3 rows
  documents: 4 rows
  usage:    47 rows

Redis:
  Keys: 128
  Memory: 2.4MB

Storage:
  Files: 12 files under storage/docs/

TOTAL: 3 collections, 54 rows, 128 keys, 12 files
```

### Step 2 — Confirmation prompt

```
This is IRREVERSIBLE. All tenant data will be permanently deleted.

Type exactly: yes wipe everything
> _
```

Read stdin. If the input is not exactly `yes wipe everything` (case-sensitive,
no extra spaces), print `Aborted.` and exit with code 1.

### Step 3 — Execute wipe in order

**3a. Qdrant — drop all collections**
```python
for collection in client.get_collections().collections:
    client.delete_collection(collection.name)
    print(f"  Dropped: {collection.name}")
```
Log: `{"event": "qdrant_wipe_complete", "collections_dropped": N}`

**3b. Postgres — truncate all tables**
```sql
TRUNCATE TABLE usage, documents, tenants CASCADE;
```
Run in a single transaction. This preserves the schema (tables, indexes,
sequences) but deletes all rows.

Log: `{"event": "postgres_wipe_complete"}`

**3c. Redis — flush all keys**
```python
redis_client.flushdb()
```
Log: `{"event": "redis_flush_complete"}`

**3d. Storage — delete all tenant directories**
```python
import shutil, os
storage_root = "storage/docs"
if os.path.exists(storage_root):
    for entry in os.scandir(storage_root):
        if entry.is_dir():
            shutil.rmtree(entry.path)
            print(f"  Deleted: {entry.path}")
```
Re-create `storage/docs/` as an empty directory.

Log: `{"event": "storage_wipe_complete"}`

### Step 4 — Verify

Re-run the audit from Step 1. All counts should be 0.

Print:
```
=== RESET COMPLETE ===
  Qdrant collections : 0 (was N)
  Postgres rows      : 0 (was N)
  Redis keys         : 0 (was N)
  Storage files      : 0 (was N)

Environment is clean. Run /ingest-demo to seed demo data.
```

If any count is non-zero, print a warning identifying the layer that didn't
fully reset.

## Safety notes

- Never skip the confirmation prompt even when called programmatically.
- Postgres uses TRUNCATE not DROP — schema is preserved, sequences reset.
- `storage/docs/` directory itself is preserved (only contents deleted).
