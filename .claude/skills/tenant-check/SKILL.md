---
name: tenant-check
description: >
  Audit any file or code path for multi-tenancy violations. Verifies that tenant_id
  always comes from JWT, never user input, and that all data access is properly scoped.
  Use when user says "check isolation", "audit tenant", "verify multi-tenancy",
  "tenant safe?", or invokes /tenant-check.
---

## What to audit

Given a file or code path, check every data access point:

### tenant_id sourcing
- PASS: `tenant_id: str = Depends(get_tenant)` in route signature
- PASS: `tenant_id` passed down from a function that received it via `get_tenant()`
- FAIL: `tenant_id` read from request body, query params, or URL path
- FAIL: `tenant_id` hardcoded or derived from user-supplied data

### Qdrant operations
- PASS: collection name is `f"{tenant_id}_docs"`
- FAIL: any collection name that doesn't include `tenant_id`

### Postgres queries
- PASS: every query has `WHERE tenant_id = $N` (or equivalent ORM filter)
- FAIL: any query that fetches rows without tenant scope

### Redis keys
- PASS: every key starts with `{tenant_id}:`
- FAIL: any key without the tenant prefix

### File system paths
- PASS: path is under `storage/docs/{tenant_id}/`
- FAIL: any path that doesn't include `tenant_id` as a directory segment

## Output format

For each violation found:
```
VIOLATION [severity: critical|high|medium]
File: <path>:<line>
Issue: <what's wrong>
Fix: <exact code change>
```

If no violations:
```
PASS — no tenant isolation violations found in <scope>
```

Always end with: total violations found, files checked.
