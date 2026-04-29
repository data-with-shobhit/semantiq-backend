---
name: check-isolation
description: CRITICAL tenant isolation test. Creates two throwaway tenants, runs 10 canonical queries each, verifies zero chunk leakage between tenants. Hard-fails on any leak. Trigger on "check isolation", "test isolation", "verify no data leak", "isolation test", "/check-isolation".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# check-isolation

The most important safety test in the platform. Proves that no chunk from
Tenant A can ever appear in Tenant B's query results.

## Invocation

```
/check-isolation
```

No arguments. Creates and destroys its own throwaway tenants.

## Exit codes

- `0` — all 10 queries × 2 tenants passed, no leakage detected
- `1` — at least one isolation violation detected (hard fail)

## Playbook

### Step 1 — Create two throwaway tenants

```python
TENANT_A = "isolation-test-financial-" + uuid4().hex[:6]
TENANT_B = "isolation-test-technical-" + uuid4().hex[:6]
```

Use `ingestion/model_registry.py` — never hardcode model IDs or dims.

```python
from ingestion.model_registry import MODEL_REGISTRY
```

Provision both (same logic as `new-tenant` skill):
- Tenant A: domain=`financial` → FinBERT, 768-dim (from model_registry)
- Tenant B: domain=`technical` → CodeBERT, 768-dim (from model_registry)

Both collections use 768-dim vectors. The isolation boundary is the collection
name (`{tenant_id}_docs`), NOT the vector space — both domains use 768-dim BERT
variants but their collection names are disjoint, enforcing physical isolation.

### Step 2 — Ingest distinct documents

**Tenant A** — ingest a small financial PDF (use `data/tesla-2023-10k.pdf` or a
5-page excerpt). Include distinctive marker phrases in metadata:
`tenant_marker: "FINANCIAL_TENANT_A_MARKER_XK9Q"`

**Tenant B** — ingest a small technical PDF (use `data/fastapi-docs.pdf` or a
5-page excerpt). Include distinctive marker phrases in metadata:
`tenant_marker: "TECHNICAL_TENANT_B_MARKER_XK9Q"`

Wait for both ingestion jobs to complete (same polling logic as `ingest-demo`).

Print: `Ingestion complete: A=<N> chunks, B=<M> chunks`

### Step 3 — Define 10 canonical queries

```python
QUERIES_A = [
    "What was the total revenue?",
    "What are the main risk factors?",
    "Describe the capital expenditure plan",
    "What was gross margin?",
    "How many employees are there?",
]

QUERIES_B = [
    "How do you define a path operation?",
    "What is dependency injection?",
    "How do you handle request body validation?",
    "What is OpenAPI schema generation?",
    "How do background tasks work?",
]
```

### Step 4 — Run all queries and collect retrieved chunk IDs

For each tenant, run all 10 queries through the full retrieval pipeline
(up to and including reranking, BEFORE the LLM generation step):

```python
def get_retrieved_chunk_ids(tenant_id: str, query: str) -> set[str]:
    # Run up to reranker, return set of Qdrant point IDs
    chunks = run_retrieval_pipeline(tenant_id, query)
    return {c.id for c in chunks}
```

Collect:
- `chunks_from_a`: union of all chunk IDs returned for any query against Tenant A
- `chunks_from_b`: union of all chunk IDs returned for any query against Tenant B

Also collect:
- `all_a_chunk_ids`: all point IDs in the `{TENANT_A}_docs` collection
- `all_b_chunk_ids`: all point IDs in the `{TENANT_B}_docs` collection

### Step 5 — Violation check

```python
# B chunks appearing in A's results
b_in_a = chunks_from_a & all_b_chunk_ids
# A chunks appearing in B's results
a_in_b = chunks_from_b & all_a_chunk_ids

violations = []
if b_in_a:
    violations.append(f"LEAK: {len(b_in_a)} Tenant B chunk(s) appeared in Tenant A results")
if a_in_b:
    violations.append(f"LEAK: {len(a_in_b)} Tenant A chunk(s) appeared in Tenant B results")
```

Also check cross-tenant score: run 5 of Tenant A's queries against Tenant B's
collection directly (this should never happen in prod but tests the Qdrant
collection name isolation):
```python
# Try to search B's collection using A's query
# If Qdrant returns B's chunks for a query run "as A", that's a config bug
```

### Step 6 — Print results

```
=== ISOLATION TEST RESULTS ===

Tenant A (financial): <TENANT_A>
Tenant B (technical): <TENANT_B>

Query results:
  Query                              A chunks  B chunks  Leak?
  ---------------------------------  --------  --------  -----
  "What was the total revenue?"      5         0         OK
  "How do you define a path op.?"    0         5         OK
  ...

Cross-tenant check:
  B chunks in A results: 0   OK
  A chunks in B results: 0   OK

=== RESULT: PASS (0 violations) ===
```

If violations found:
```
=== RESULT: FAIL ===

VIOLATIONS DETECTED:
  1. Query "What was the total revenue?" against Tenant A returned chunk ID <uuid>
     which belongs to Tenant B (collection: {TENANT_B}_docs)
     Chunk text: "<first 100 chars>"

IMMEDIATE ACTION REQUIRED: Tenant isolation is broken.
Check: Qdrant collection name derivation in retrieval/hybrid.py
```

### Step 7 — Cleanup (always runs, even on failure)

Call teardown sequence for both `TENANT_A` and `TENANT_B` without confirmation
prompt.

Log: `{"event": "isolation_test_complete", "result": "pass/fail", "violations": N}`

### Step 8 — Exit

```python
import sys
sys.exit(0 if not violations else 1)
```
