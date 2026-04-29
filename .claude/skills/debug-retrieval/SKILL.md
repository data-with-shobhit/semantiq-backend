---
name: debug-retrieval
description: Print the full retrieval waterfall for a query: HyDE rewrite, SPLADE sparse hits (BM25 fallback), dense hits, RRF fusion, reranker scores, threshold decision, final chunks. Trigger on "debug retrieval", "why did retrieval fail", "show retrieval waterfall", "/debug-retrieval".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# debug-retrieval

Prints every stage of the retrieval pipeline for a given tenant and query, so
you can see exactly why a particular chunk was or wasn't returned.

## Invocation

```
/debug-retrieval <tenant_id> "<query>"
```

## Playbook

IMPORTANT: `tenant_id` is provided here as a debug argument. In production code,
`tenant_id` always comes from the JWT. This skill is a developer tool only.

### Step 0 — Header

Fetch tenant's domain and embed config from Postgres:
```sql
SELECT domain, embed_model, embed_dim FROM tenants WHERE id = $1;
```

```
=== RETRIEVAL WATERFALL ===
Tenant  : <tenant_id>
Domain  : <domain>
Model   : <embed_model>  (dim=<embed_dim>)
Query   : <query>
Time    : <ISO timestamp>
```

### Step 1 — Original query

Print as-is.

### Step 2 — HyDE rewrite

HyDE generates a hypothetical answer via LM Studio/Gemma, then embeds it using
the tenant's domain-specific model (NOT a generic model):

```python
from retrieval.rewriter import hyde_rewrite
from ingestion.embedder import Embedder
from llm.lmstudio_backend import LMStudioBackend

llm = LMStudioBackend()
hyde_doc = await hyde_rewrite(query, llm)

embedder = Embedder(domain)                   # uses tenant's domain model
hyde_embedding = embedder.encode_one(hyde_doc)
```

Print:
```
[HyDE Hypothetical Document]
  "<first 300 chars of hyde_doc>..."
  (embedding via <embed_model>, dim=<embed_dim>)
```

### Step 3 — SPLADE sparse encoding

SPLADE (`naver/splade-v3`) handles implicit query expansion for all domains.
OOV caveat: technical/code tokens may be under-weighted; BM25 fallback handles those.

```python
from retrieval import splade as splade_enc
indices, values = splade_enc.encode(query)
```

Print:
```
[SPLADE Encoding]
  Non-zero terms : <len(indices)>
  Top-5 terms    : <decode top 5 tokens by weight>
  Status         : splade / bm25_fallback
```

If `indices` is empty: SPLADE failed or OOV — BM25 fallback will be used.

### Step 4 — Sparse hits (top 10)

SPLADE primary, BM25 fallback over dense hits if SPLADE returns empty:

```python
from retrieval.hybrid import _sparse_search, _dense_search
# BM25 fallback uses rank-bm25 over the dense hit list
```

Print table:
```
[Sparse Hits — top 10]  (method: splade / bm25_fallback)
  Rank  Score    doc_id  chunk_idx  Text preview (100 chars)
  ----  -------  ------  ---------  ------------------------
  1     0.8432   42      7          "...revenue increased by..."
  2     0.7801   42      8          "...operating margin was..."
  ...
```

### Step 5 — Dense hits (top 10)

Search Qdrant using the HyDE embedding vector:
```python
from qdrant_client import QdrantClient
results = client.search(
    collection_name=f"{tenant_id}_docs",
    query_vector=hyde_embedding,
    limit=10,
)
```

Print table (same format as BM25 but with cosine similarity scores).

### Step 6 — RRF fusion

```python
from retrieval.hybrid import rrf_fuse
fused = rrf_fuse(bm25_hits, dense_hits, k=60)
```

Print:
```
[RRF Fusion — top 10]
  Rank  RRF Score  BM25 rank  Dense rank  chunk_id
  ----  ---------  ---------  ----------  --------
  1     0.03271    1          2           uuid-xxx
  2     0.03158    3          1           uuid-yyy
  ...
```

Note chunks that appeared in only one retriever with: `(BM25 only)` or `(dense only)`.

### Step 7 — CrossEncoder reranking (top 5)

```python
from retrieval.reranker import CrossEncoderReranker
reranker = CrossEncoderReranker()
reranked = reranker.rerank(query, fused[:20])
```

Print:
```
[Reranker — top 5]
  Rank  CE Score  delta_rank  Text preview (150 chars)
  ----  --------  ----------  ------------------------
  1     0.9123    +0          "..."
  2     0.8854    -1          "..."
  3     0.7201    +3          "..."
  4     0.6998    -2          "..."
  5     0.4312    +8          "..."
```

`delta_rank` = RRF rank minus reranker rank (positive = moved up).

### Step 8 — Threshold gate

```
[Threshold Gate]
  Top score  : <value>
  Threshold  : 0.72
  Decision   : PASS / FAIL
```

If FAIL: `→ Pipeline would return: "I don't have that information."`
If PASS: `→ Chunks 1-5 will be sent to LLM.`

### Step 9 — Final chunks (what the LLM receives)

For each of the top-5 chunks (or fewer if FAIL), print:
```
[Source 1 | score: 0.91]
  doc_id: 42  chunk_idx: 7  page: 12
  fiscal_year: 2023  section: income_statement
  "Full chunk text here..."
```

### Step 10 — Footer

```
=== END WATERFALL ===
Latency breakdown:
  HyDE rewrite  : Xms
  BM25 search   : Xms
  Dense search  : Xms
  RRF fusion    : Xms
  Reranking     : Xms
  Total         : Xms
```
