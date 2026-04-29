---
name: update-checklist
description: Read CLAUDE.md build order, inspect git log since last update, auto-mark completed items based on merged commits, show diff before writing. Trigger on "update checklist", "mark complete", "sync checklist", "update progress", "/update-checklist".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# update-checklist

Keeps the CLAUDE.md build order checklist in sync with actual git history.
Reads commit messages, identifies which checklist items are done, and marks them.

## Invocation

```
/update-checklist
```

## Playbook

### Step 1 — Read current checklist

```python
with open("CLAUDE.md") as f:
    content = f.read()

# Find the build order section
import re
section_match = re.search(
    r"(## Build order \(checklist\).*?)(?=\n## |\Z)",
    content,
    re.DOTALL
)
checklist_section = section_match.group(1)

# Extract items
items = re.findall(r"- \[([ x])\] (.+)", checklist_section)
# items = [('x', 'docker-compose scaffold'), (' ', 'Postgres schema'), ...]
```

Print: `Found <N> checklist items (<K> already complete, <M> remaining)`

### Step 2 — Get git log since last checklist update

Find the last commit that touched CLAUDE.md:
```bash
git log --oneline -- CLAUDE.md | head -1
```

Then get all commits since then:
```bash
git log --oneline --since="<last_claude_md_commit_date>"
```

Also get all commit messages for analysis:
```bash
git log --format="%H %s%n%b" | head -200
```

### Step 3 — Map commits to checklist items

For each unchecked item, determine if a commit likely implements it.
Use keyword matching between commit messages and checklist item text:

```python
KEYWORD_MAP = {
    "docker-compose scaffold": ["docker-compose", "compose", "scaffold"],
    "Postgres schema": ["postgres", "schema", "migration", "CREATE TABLE"],
    "JWT auth": ["jwt", "auth", "get_tenant", "middleware"],
    "POST /signup": ["signup", "provision", "tenant.*create"],
    "LocalStorage backend": ["storage", "LocalStorage", "backend"],
    "Chunker + metadata extractor": ["chunk", "chunker", "extractor", "metadata"],
    "Celery ingestion task": ["celery", "task", "ingest", "embed.*upsert"],
    "POST /ingest": ["/ingest", "ingest.*route", "upload.*enqueue"],
    "Synonym expansion + HyDE rewriter": ["hyde", "synonym", "rewriter", "expand"],
    "Hybrid retrieval": ["hybrid", "bm25", "rrf", "dense.*retriev"],
    "CrossEncoder reranker": ["rerank", "crossencoder", "cross.encoder"],
    "LangGraph ReAct query agent": ["langgraph", "react.*agent", "query.*graph"],
    "POST /query": ["/query", "query.*route"],
    "Streamlit demo UI": ["streamlit", "demo.*ui", "ui.*demo"],
    "test_isolation.py": ["isolation", "test_isolation"],
    "RAGAS eval harness": ["ragas", "eval.*harness", "faithfulness"],
    "LangSmith tracing": ["langsmith", "tracing", "trace"],
    "Prometheus metrics": ["prometheus", "metrics", "grafana"],
    "README": ["readme", "architecture.*diagram", "loom"],
}

newly_completed = []
for item_text in unchecked_items:
    for pattern_key, keywords in KEYWORD_MAP.items():
        if pattern_key.lower() in item_text.lower():
            for kw in keywords:
                if any(re.search(kw, msg, re.IGNORECASE) for msg in commit_messages):
                    newly_completed.append(item_text)
                    break
```

### Step 4 — Show diff before writing

Print:
```
Proposed checklist changes:

  Items to mark complete (based on git log):
    ✓ docker-compose scaffold (Postgres + Redis + Qdrant)
      → matched commit: "feat: add docker-compose.yml with postgres, redis, qdrant"
    ✓ JWT auth + get_tenant() dependency
      → matched commit: "feat(auth): implement JWT middleware and get_tenant dep"

  Items remaining (no matching commits found):
    ○ Postgres schema (tenants, documents, usage)
    ○ POST /signup — provisions Postgres row + empty Qdrant collection
    ...

  Total: 2 newly completed, <N> remaining
```

### Step 5 — Prompt for confirmation

```
Apply these changes to CLAUDE.md? [y/N]: _
```

If user types `y` or `yes`: proceed.
If user types `n`, `no`, or presses Enter: print `Aborted. No changes made.` and exit.

### Step 6 — Write updated CLAUDE.md

Use a targeted replacement: only update the `[ ]` → `[x]` markers for the
newly-completed items. Do not touch any other content in CLAUDE.md.

```python
updated_section = checklist_section
for item_text in newly_completed:
    # Replace '- [ ] item_text' with '- [x] item_text'
    updated_section = updated_section.replace(
        f"- [ ] {item_text}",
        f"- [x] {item_text}",
    )

updated_content = content.replace(checklist_section, updated_section)
with open("CLAUDE.md", "w") as f:
    f.write(updated_content)
```

Print: `CLAUDE.md updated. <N> items marked complete.`

### Step 7 — Append newly-added items

If the user passes `--add "New item text"`, append it to the checklist as an
unchecked item at the end of the build order section.

### Step 8 — Summary

```
Checklist status:
  Complete  : <K> / <total> items
  Remaining : <M> items
  Progress  : <pct>%

Next up: "<first unchecked item>"
```
