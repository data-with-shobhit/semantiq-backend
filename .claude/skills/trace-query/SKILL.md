---
name: trace-query
description: Run the LangGraph ReAct agent with full node-level tracing. Prints each node's input state, tool calls, and output state. Trigger on "trace query", "trace agent", "show agent steps", "langgraph trace", "/trace-query".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# trace-query

Runs a query through the full LangGraph ReAct agent with verbose node-level
tracing so you can see exactly what the agent decided at each step.

## Invocation

```
/trace-query <tenant_id> "<query>"
```

## Playbook

IMPORTANT: `tenant_id` is a debug argument here. In production it comes from the JWT.

### Step 1 — Header

```
=== LANGGRAPH TRACE ===
Tenant : <tenant_id>
Query  : <query>
Time   : <ISO timestamp>
```

### Step 2 — Configure tracing

Enable LangSmith tracing if available (no LangChain required — LangGraph sends
traces directly when `LANGCHAIN_TRACING_V2=true`):
```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = f"rag-trace-{tenant_id}"
```

**Do NOT use `langchain_core.callbacks.BaseCallbackHandler`** — we have no
LangChain dependency. Use LangGraph's native `stream_events` API instead.

### Step 3 — Run the agent with stream_events

```python
from graph.query_graph import build_query_graph
import time

graph = build_query_graph(tenant_id)
node_timings = {}
result = None

# LangGraph stream_events emits one dict per state transition
async for event in graph.astream_events(
    {"question": query, "tenant_id": tenant_id},
    version="v2",
):
    kind = event["event"]
    name = event.get("name", "")
    data = event.get("data", {})

    if kind == "on_chain_start" and name in ("rewriter", "retriever", "reranker", "threshold_gate", "generator"):
        node_timings[name] = {"start": time.perf_counter(), "input": data.get("input", {})}
        print(f"\n{'='*60}")
        print(f"NODE START: {name}")
        for k, v in node_timings[name]["input"].items():
            print(f"  {k}: {str(v)[:200]}")

    elif kind == "on_chain_end" and name in node_timings:
        elapsed = int((time.perf_counter() - node_timings[name]["start"]) * 1000)
        node_timings[name]["latency"] = elapsed
        node_timings[name]["output"] = data.get("output", {})
        print(f"NODE END: {name}  ({elapsed}ms)")
        for k, v in node_timings[name]["output"].items():
            print(f"  {k}: {str(v)[:200]}")

    elif kind == "on_tool_start":
        print(f"\n  TOOL CALL: {name}")
        print(f"  INPUT    : {str(data.get('input', ''))[:300]}")

    elif kind == "on_tool_end":
        print(f"  RESULT   : {str(data.get('output', ''))[:300]}")

# Run synchronously in non-async context:
import asyncio
asyncio.run(run_trace())
result = graph.invoke({"question": query, "tenant_id": tenant_id})
```

Generator node calls `OllamaBackend.generate()` via `llm/ollama_backend.py`.
No external API. No Anthropic SDK. Runs fully local via Ollama on port 11434.

### Step 4 — Node summary table

After execution, print a summary of the traversal path:

```
[Agent Traversal Summary]
  Step  Node           Action taken                     Latency
  ----  -------------  -------------------------------  -------
  1     rewriter       HyDE + synonym expansion         45ms
  2     retriever      retrieve("query", filters={})    120ms
  3     reranker       CrossEncoder top-5               85ms
  4     threshold_gate PASS (top score: 0.87)           1ms
  5     generator      LLM call (512 tokens in/out)     1200ms

  Total agent latency: 1451ms
  LangSmith trace: https://smith.langchain.com/...
```

### Step 5 — Final answer

```
[Final Answer]
<answer text>

[Cited Sources]
  [Source 1] doc_id=42, chunk_idx=7, score=0.87
  [Source 2] doc_id=42, chunk_idx=8, score=0.81
  ...
```

### Step 6 — Multi-hop detection

If the agent called `retrieve` more than once, print:
```
[Multi-hop detected — <N> retrieval calls]
  Hop 1: query="<first query>"     → <K> chunks
  Hop 2: query="<second query>"    → <K> chunks
  Reason for second hop: <agent reasoning from trace>
```

## Error handling

- LangSmith not configured → print a note and continue without remote tracing
- Ollama not running → `ConnectionRefusedError` on port 11434. Print: `Ollama unreachable. Run: ollama serve` then verify `ollama list` shows `gemma-rag`
- `gemma-rag` model missing → print: `Run: ollama create gemma-rag -f Modelfile.gemma-rag`
- Agent hits max iterations → print the last state and what the agent was trying
- Tool call fails → print the error and the tool's input that caused it
