---
name: teardown-tenant
description: Delete a tenant completely. Removes Qdrant collection, Redis keys, storage directory, documents rows, usage rows, and tenants row in safe FK order. Trigger on "teardown tenant", "delete tenant", "remove tenant", "/teardown-tenant".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# teardown-tenant

Completely and irreversibly deletes a tenant and all associated data.

## Invocation

```
/teardown-tenant <tenant_id>
```

## Playbook

### Step 1 — Count before deleting (confirmation)

Query all layers and print counts:

```sql
SELECT
  (SELECT COUNT(*) FROM documents WHERE tenant_id = $1) AS documents,
  (SELECT COUNT(*) FROM usage     WHERE tenant_id = $1) AS usage_events,
  (SELECT COUNT(*) FROM tenants   WHERE id        = $1) AS tenant_row;
```

```python
# Qdrant
try:
    info = client.get_collection(f"{tenant_id}_docs")
    vector_count = info.vectors_count
except:
    vector_count = 0

# Redis
redis_count = len(redis_client.keys(f"{tenant_id}:*"))

# Storage
import os
storage_path = f"storage/docs/{tenant_id}"
file_count = sum(len(f) for _, _, f in os.walk(storage_path)) if os.path.exists(storage_path) else 0
```

Print:

```
About to delete tenant: <tenant_id>

  Qdrant vectors  : <N>  (collection: {tenant_id}_docs)
  Redis keys      : <N>  (pattern: {tenant_id}:*)
  Storage files   : <N>  (path: storage/docs/{tenant_id}/)
  Document rows   : <N>
  Usage event rows: <N>
  Tenant row      : <1 or 0>

Type the tenant_id to confirm deletion: _
```

Wait for user confirmation. If the typed value does not match `tenant_id` exactly,
abort with: `Aborted — tenant_id mismatch.`

If tenant row count is 0, warn: `Tenant not found in Postgres. Proceeding to clean up orphaned data.`

### Step 2 — Delete in safe order

**2a. Qdrant collection**
```python
client.delete_collection(f"{tenant_id}_docs")
# Ignore NotFoundException
```
Log: `{"event": "qdrant_collection_deleted", "tenant_id": tenant_id}`

**2b. Redis keys**
```python
keys = redis_client.keys(f"{tenant_id}:*")
if keys:
    redis_client.delete(*keys)
```
Log: `{"event": "redis_keys_purged", "tenant_id": tenant_id, "count": len(keys)}`

**2c. Storage directory**
```python
import shutil
if os.path.exists(f"storage/docs/{tenant_id}"):
    shutil.rmtree(f"storage/docs/{tenant_id}")
```
Log: `{"event": "storage_deleted", "tenant_id": tenant_id}`

**2d. Postgres rows (FK-safe order)**
```sql
DELETE FROM documents WHERE tenant_id = $1;
DELETE FROM usage     WHERE tenant_id = $1;
DELETE FROM tenants   WHERE id        = $1;
```
All three in a single transaction. Log row counts deleted.

### Step 3 — Verify

Re-run the Step 1 counts query. All should be 0. Print:

```
Teardown complete for tenant: <tenant_id>
  All counts verified at zero.
```

If any count is non-zero, print a warning and the remaining count so the user
can investigate.

## Error handling

- Qdrant 404 on delete → log and continue (orphaned Postgres row case)
- Redis unavailable → log warning, continue with Postgres/storage cleanup
- Postgres constraint error → print the SQL error; likely a missing FK cascade
