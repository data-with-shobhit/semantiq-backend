---
name: eval-rag
description: Run RAGAS evaluation for a tenant. Reads held-out questions from eval/questions/{tenant_id}.json, runs the full query pipeline, computes faithfulness, context precision, and answer relevance. Writes scorecard. Trigger on "eval rag", "evaluate rag", "ragas eval", "run evaluation", "/eval-rag".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# eval-rag

Runs RAGAS evaluation against a tenant's held-out question set and writes a
timestamped scorecard.

## Invocation

```
/eval-rag <tenant_id>
```

## Playbook

### Step 1 — Load held-out questions

```python
import json, os

questions_path = f"eval/questions/{tenant_id}.json"
if not os.path.exists(questions_path):
    print(f"No held-out questions found at {questions_path}")
    print("Create this file with the structure:")
    print(json.dumps([{"question": "...", "ground_truth": "..."}], indent=2))
    exit(1)

with open(questions_path) as f:
    questions = json.load(f)

print(f"Loaded {len(questions)} questions for tenant {tenant_id}")
```

Expected format:
```json
[
  {
    "question": "What was Tesla's total revenue in 2023?",
    "ground_truth": "Tesla's total revenue in 2023 was $97.69 billion."
  }
]
```

### Step 2 — Run full query pipeline for each question

For each question, run the complete pipeline and capture:
- `answer`: the LLM's generated answer
- `contexts`: list of retrieved chunk texts (up to 5)
- `ground_truth`: from the question file

```python
from graph.query_graph import build_query_graph

results = []
graph = build_query_graph(tenant_id)

for i, q in enumerate(questions):
    print(f"Running {i+1}/{len(questions)}: {q['question'][:60]}...")
    raw = graph.invoke({"question": q["question"], "tenant_id": tenant_id})
    results.append({
        "question": q["question"],
        "answer": raw["answer"],
        "contexts": [c.text for c in raw["chunks"]],
        "ground_truth": q["ground_truth"],
    })
```

Log each run: `{"event": "eval_query", "tenant_id": tenant_id, "question_idx": i}`

### Step 3 — Compute RAGAS metrics

```python
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy
from datasets import Dataset

dataset = Dataset.from_list(results)
scores = evaluate(
    dataset,
    metrics=[faithfulness, context_precision, answer_relevancy],
)
```

Print live progress as each metric is computed.

### Step 4 — Print scorecard table

```
=== RAGAS SCORECARD ===
Tenant   : <tenant_id>
Questions: <N>
Run at   : <ISO timestamp>

┌──────────────────────┬────────┬──────────┐
│ Metric               │ Score  │ Status   │
├──────────────────────┼────────┼──────────┤
│ Faithfulness         │ 0.8923 │ OK       │
│ Context Precision    │ 0.7641 │ OK       │
│ Answer Relevance     │ 0.9012 │ OK       │
└──────────────────────┴────────┴──────────┘

Per-question breakdown:
  Q1  F=0.91  CP=0.78  AR=0.95  "What was Tesla's total revenue..."
  Q2  F=0.85  CP=0.72  AR=0.88  "What are the main risk factors..."
  ...
```

If any metric is below 0.70, print: `WARNING: <metric> below threshold (0.70)`
If any metric dropped >5% from the last run, print: `REGRESSION: <metric> dropped X% from last run`

### Step 5 — Write results to file

```python
import time, json

timestamp = int(time.time())
output_path = f"eval/runs/{timestamp}_{tenant_id}.json"
os.makedirs("eval/runs", exist_ok=True)

output = {
    "tenant_id": tenant_id,
    "timestamp": timestamp,
    "question_count": len(questions),
    "scores": {
        "faithfulness": float(scores["faithfulness"]),
        "context_precision": float(scores["context_precision"]),
        "answer_relevancy": float(scores["answer_relevancy"]),
    },
    "per_question": results,
}
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults written to: {output_path}")
```

### Step 6 — Compare with previous run

Find the most recent previous run for this tenant:
```python
prev_runs = sorted(glob.glob(f"eval/runs/*_{tenant_id}.json"))
```

If a previous run exists, load it and print delta table:
```
Comparison vs previous run (<prev_timestamp>):
  Faithfulness     : 0.8923 vs 0.8812  (+0.0111)  ↑
  Context Precision: 0.7641 vs 0.7890  (-0.0249)  ↓  REGRESSION
  Answer Relevance : 0.9012 vs 0.8991  (+0.0021)  →
```

Regression = delta < -0.05 on any metric.

## Error handling

- RAGAS API key not set → print setup instructions
- LLM call fails → retry once; on second failure, log and skip that question
- All questions fail → abort and print the common error
