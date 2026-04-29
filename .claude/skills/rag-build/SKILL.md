---
name: rag-build
description: >
  Build the next unchecked item from the RAG platform build order in CLAUDE.md.
  Implements the component end-to-end, writes tests alongside, verifies it works,
  then updates the checklist. Use when user says "build next", "next step",
  "implement next item", or invokes /rag-build.
---

## Steps

1. Read `CLAUDE.md` — find first unchecked `[ ]` item in the build order.
2. Read `ARCHITECTURE.md` — understand the design for that component.
3. Identify which files need to be created or modified (follow the project structure in CLAUDE.md).
4. Write the implementation:
   - Python 3.11+, full type hints
   - Async in API layer, sync in Celery workers
   - `structlog` not `print`, every log includes `tenant_id`
   - Functions ≤ 50 lines — split if longer
   - Multi-tenancy rules: `tenant_id` from JWT only, never user input
5. Write or update the corresponding test file (mirror source tree).
6. Verify end-to-end — describe how to test it manually or run the test.
7. Update the CLAUDE.md checklist: change `[ ]` to `[x]` for the completed item.

## Multi-tenancy checklist (apply to every component)

- Qdrant collection: `{tenant_id}_docs`
- Postgres queries: `WHERE tenant_id = ?`
- Redis keys: `{tenant_id}:*`
- File paths: `storage/docs/{tenant_id}/`
- `tenant_id` source: JWT `sub` claim via `get_tenant()` dependency only

## Output format

After completing the build:
- List files created/modified
- State how to verify the component works
- Confirm the CLAUDE.md checklist is updated
