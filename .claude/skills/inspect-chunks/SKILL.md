---
name: inspect-chunks
description: Dump every chunk for a document: chunk_idx, text preview, metadata payload, and 3 nearest neighbor chunks by cosine similarity. Trigger on "inspect chunks", "show chunks", "dump chunks", "chunk details", "/inspect-chunks".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# inspect-chunks

Dumps all chunks for a specific document and shows nearest-neighbor relationships
within the same Qdrant collection.

## Invocation

```
/inspect-chunks <tenant_id> <doc_id>
```

## Playbook

IMPORTANT: `tenant_id` is a debug argument. In production it comes from the JWT.

### Step 1 — Header

```
=== CHUNK INSPECTOR ===
Tenant : <tenant_id>
doc_id : <doc_id>
```

Verify the document exists:
```sql
SELECT filename, status, chunk_count, uploaded_at
FROM documents
WHERE id = $1 AND tenant_id = $2;
```

If not found: `Error: doc_id <N> not found for tenant <tenant_id>`. Exit.

Print:
```
Document: <filename>
Status  : <status>
Chunks  : <chunk_count>
Uploaded: <uploaded_at>
```

### Step 2 — Fetch all chunks from Qdrant

```python
# Scroll through all points for this doc_id
points, _ = client.scroll(
    collection_name=f"{tenant_id}_docs",
    scroll_filter=Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    ),
    limit=1000,
    with_vectors=True,
    with_payload=True,
)
# Sort by chunk_idx
points.sort(key=lambda p: p.payload.get("chunk_idx", 0))
```

### Step 3 — Print each chunk

For each chunk:
```
────────────────────────────────────────────────────
Chunk #<chunk_idx>  (id: <uuid>)
────────────────────────────────────────────────────
Metadata:
  doc_type   : <value>
  fiscal_year: <value>
  section    : <value>
  has_numbers: <value>
  page       : <value>

Text (200 chars):
  "<first 200 characters of text>..."

Full text length: <N> characters / ~<M> tokens
```

### Step 4 — 3 nearest neighbors per chunk

For each chunk, find 3 nearest neighbors within the same collection (excluding
itself and other chunks from the same doc_id):

```python
neighbors = client.search(
    collection_name=f"{tenant_id}_docs",
    query_vector=point.vector,
    limit=4,  # +1 to exclude self
    query_filter=Filter(
        must_not=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    ),
)
```

Print after each chunk's text:
```
Nearest neighbors (other documents):
  1. score=0.94  doc_id=<X>  chunk_idx=<Y>  "<first 80 chars>..."
  2. score=0.91  doc_id=<X>  chunk_idx=<Z>  "<first 80 chars>..."
  3. score=0.88  doc_id=<W>  chunk_idx=<Q>  "<first 80 chars>..."
```

If no neighbors found across other documents, print: `(no cross-document neighbors)`

### Step 5 — Summary statistics

After all chunks:
```
=== SUMMARY ===
Total chunks  : <N>
Avg text len  : <X> chars
Has-numbers   : <K> chunks (<pct>%)
Sections found: <list of unique sections>
Pages covered : <min>–<max>

Nearest-neighbor stats:
  Avg top-1 similarity: <value>
  Min top-1 similarity: <value>  (chunk #<idx> — possibly isolated)
  Max top-1 similarity: <value>  (chunk #<idx> — possible duplicate)
```

If `max top-1 similarity > 0.98`, warn: `Possible near-duplicate chunks detected at indices: <list>`

### Step 6 — Chunk gaps detection

Check for missing `chunk_idx` values (gaps indicate failed upsert):
```python
expected = set(range(len(points)))
actual = {p.payload["chunk_idx"] for p in points}
gaps = expected - actual
if gaps:
    print(f"WARNING: Missing chunk indices: {sorted(gaps)}")
```
