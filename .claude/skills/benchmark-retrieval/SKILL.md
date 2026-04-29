---
name: benchmark-retrieval
description: Compare current retrieval against a baseline config. Reports Recall@5, MRR, and nDCG side-by-side with deltas and flags regressions. Trigger on "benchmark retrieval", "compare retrieval", "retrieval regression", "/benchmark-retrieval".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# benchmark-retrieval

Runs the held-out question set against both current code and a baseline config,
then reports retrieval metrics side-by-side.

## Invocation

```
/benchmark-retrieval <config_name>
```

`config_name` refers to a file at `eval/baselines/<config_name>.json`.

## Playbook

### Step 1 — Load baseline

```python
import json, os

baseline_path = f"eval/baselines/{config_name}.json"
if not os.path.exists(baseline_path):
    print(f"Baseline not found: {baseline_path}")
    print("Available baselines:")
    for f in os.listdir("eval/baselines"):
        print(f"  {f}")
    exit(1)

with open(baseline_path) as f:
    baseline = json.load(f)
```

Baseline format:
```json
{
  "name": "rrf-k60-threshold-0.72",
  "description": "Default config as of 2026-04-01",
  "tenant_id": "acme-corp",
  "questions": [
    {
      "question": "...",
      "relevant_chunk_ids": ["uuid1", "uuid2", "uuid3"]
    }
  ],
  "metrics": {
    "recall_at_5": 0.82,
    "mrr": 0.74,
    "ndcg": 0.79
  }
}
```

### Step 2 — Run current retrieval

For each question in the baseline, run the current retrieval pipeline
(up to and including reranking, before LLM generation):

```python
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import CrossEncoderReranker

retriever = HybridRetriever(tenant_id=baseline["tenant_id"])
reranker = CrossEncoderReranker()

current_results = []
for q in baseline["questions"]:
    fused = retriever.retrieve(q["question"], top_k=20)
    reranked = reranker.rerank(q["question"], fused)[:5]
    returned_ids = [c.id for c in reranked]
    current_results.append({
        "question": q["question"],
        "relevant": set(q["relevant_chunk_ids"]),
        "returned": returned_ids,
    })
```

### Step 3 — Compute metrics

```python
def recall_at_k(returned: list, relevant: set, k: int = 5) -> float:
    return len(set(returned[:k]) & relevant) / max(len(relevant), 1)

def mrr(returned: list, relevant: set) -> float:
    for i, r in enumerate(returned):
        if r in relevant:
            return 1.0 / (i + 1)
    return 0.0

def ndcg_at_k(returned: list, relevant: set, k: int = 5) -> float:
    dcg = sum(
        1.0 / (i + 1)  # log2(i+2) is more standard but this is fine for k<=5
        for i, r in enumerate(returned[:k]) if r in relevant
    )
    ideal = sum(1.0 / (i + 1) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0

current_metrics = {
    "recall_at_5": sum(recall_at_k(r["returned"], r["relevant"]) for r in current_results) / len(current_results),
    "mrr": sum(mrr(r["returned"], r["relevant"]) for r in current_results) / len(current_results),
    "ndcg": sum(ndcg_at_k(r["returned"], r["relevant"]) for r in current_results) / len(current_results),
}
```

### Step 4 — Print side-by-side report

```
=== RETRIEVAL BENCHMARK ===
Config   : <config_name>
Tenant   : <tenant_id>
Questions: <N>

┌──────────────┬──────────┬─────────┬─────────┬────────┐
│ Metric       │ Baseline │ Current │ Delta   │ Status │
├──────────────┼──────────┼─────────┼─────────┼────────┤
│ Recall@5     │ 0.8200   │ 0.8450  │ +0.0250 │  ↑ OK  │
│ MRR          │ 0.7400   │ 0.7200  │ -0.0200 │  ↓ OK  │
│ nDCG         │ 0.7900   │ 0.7400  │ -0.0500 │  ↓ WARN│
└──────────────┴──────────┴─────────┴─────────┴────────┘
```

Status rules:
- Delta >= +0.01: `↑ OK` (improvement)
- -0.05 <= Delta < 0: `↓ OK` (minor drop)
- Delta < -0.05: `↓ REGRESSION` (flag)

### Step 5 — Per-question regression details

For any question where all metrics dropped (worst cases first):
```
Top regressions:
  1. "What was gross margin in Q3?" — Recall 0.60 vs 1.00  (-0.40)
     Returned: [uuid-x, uuid-y, ...]
     Missing relevant: [uuid-z]
     Note: chunk uuid-z was rank 8 in BM25 but rank 15 in dense — RRF fusion
     didn't boost it enough. Consider reducing k in RRF from 60 → 40.
```

### Step 6 — Write benchmark result

```python
output = {
    "config_name": config_name,
    "timestamp": int(time.time()),
    "baseline_metrics": baseline["metrics"],
    "current_metrics": current_metrics,
    "regressions": [r for r in current_results if ...],
}
os.makedirs("eval/benchmark_runs", exist_ok=True)
with open(f"eval/benchmark_runs/{int(time.time())}_{config_name}.json", "w") as f:
    json.dump(output, f, indent=2)
```

Print: `Results written to eval/benchmark_runs/<timestamp>_<config_name>.json`

If any regression: exit with code 1.
