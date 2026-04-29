---
name: replay-query
description: Fetch a LangSmith run by ID and re-run the exact same query against current code. Shows side-by-side diff of retrieved chunks and final answer. Trigger on "replay query", "compare langsmith run", "regression check run", "/replay-query".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# replay-query

Fetches a historical LangSmith run, re-executes the same query against current
code, and shows a diff so you can verify retrieval changes didn't regress.

## Invocation

```
/replay-query <langsmith_run_id>
```

## Playbook

### Step 1 — Fetch run from LangSmith

```python
from langsmith import Client

client = Client()
run = client.read_run(langsmith_run_id)

# Extract original inputs
original_query = run.inputs.get("question")
original_tenant_id = run.inputs.get("tenant_id")

# Extract original outputs
original_answer = run.outputs.get("answer", "")
original_chunks = run.outputs.get("chunks", [])
# or extract from child runs if stored at node level
```

Print:
```
=== REPLAY QUERY ===
LangSmith Run: <langsmith_run_id>
Original run : <run.start_time>
Tenant       : <original_tenant_id>
Query        : "<original_query>"
```

If LangSmith not configured or run not found:
```
Error: Could not fetch run <id>
Check LANGCHAIN_API_KEY and LANGCHAIN_PROJECT env vars.
```

### Step 2 — Re-run against current code

```python
from graph.query_graph import build_query_graph

graph = build_query_graph(original_tenant_id)
current_result = graph.invoke({
    "question": original_query,
    "tenant_id": original_tenant_id,
})

current_answer = current_result["answer"]
current_chunks = current_result["chunks"]
```

### Step 3 — Chunk diff

Compare chunk IDs and scores between original and current:

```python
original_ids = {c["id"]: c for c in original_chunks}
current_ids  = {c.id: c for c in current_chunks}

added   = set(current_ids) - set(original_ids)
removed = set(original_ids) - set(current_ids)
kept    = set(original_ids) & set(current_ids)
```

Print:
```
[Chunk Diff]
  Original: <N> chunks  →  Current: <M> chunks

  KEPT (<K> chunks):
    [Source 1] id=<uuid>  score: 0.89 → 0.91  (Δ +0.02)  "<80 chars>"
    [Source 2] id=<uuid>  score: 0.84 → 0.80  (Δ -0.04)  "<80 chars>"

  ADDED in current (<N> chunks):
    + id=<uuid>  score: 0.78  "<80 chars>"

  REMOVED from original (<N> chunks):
    - id=<uuid>  was score: 0.82  "<80 chars>"
```

### Step 4 — Answer diff

Use unified diff format:

```python
import difflib

diff = list(difflib.unified_diff(
    original_answer.splitlines(keepends=True),
    current_answer.splitlines(keepends=True),
    fromfile="original",
    tofile="current",
    lineterm="",
))
```

Print:
```
[Answer Diff]
--- original (<run.start_time>)
+++ current  (<now>)
@@ ... @@
  Unchanged line
- Removed line
+ Added line
```

If answer is identical: `[Answer Diff] No change — answers are identical`

### Step 5 — Regression assessment

Compute a similarity score between answers:
```python
from difflib import SequenceMatcher
similarity = SequenceMatcher(None, original_answer, current_answer).ratio()
```

Print:
```
[Regression Assessment]
  Answer similarity: <score> (1.0 = identical)
  Chunks added     : <N>
  Chunks removed   : <N>
  Score delta (avg): <Δ>

  Verdict: OK / REVIEW NEEDED / REGRESSION
```

Verdict rules:
- similarity >= 0.90 and no chunks removed: `OK`
- similarity 0.70–0.90 or 1-2 chunks removed: `REVIEW NEEDED`
- similarity < 0.70 or >2 chunks removed: `REGRESSION`

### Step 6 — LangSmith link

Print the URL for the original run so the user can open it:
```
Original LangSmith trace: https://smith.langchain.com/...
```
