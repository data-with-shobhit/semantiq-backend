---
name: tail-logs
description: Stream structured logs from API and Celery worker filtered to a specific tenant_id. Pretty-prints JSON with color-coded log levels. Trigger on "tail logs", "stream logs", "show logs", "filter logs", "/tail-logs".
allowed-tools: Bash, Read, Write, Edit, Glob, Grep
---

# tail-logs

Streams structlog JSON output from both the FastAPI API process and the Celery
worker, filtered to a specific tenant.

## Invocation

```
/tail-logs <tenant_id>
```

## Playbook

### Step 1 — Locate log sources

The platform uses structlog with JSON output. Logs may come from:

1. **API process** — stdout of the uvicorn process or a log file at `logs/api.log`
2. **Celery worker** — stdout of the celery worker or `logs/worker.log`
3. **Docker logs** — if running via docker-compose, use `docker-compose logs -f api worker`

Detect which log source is available:
```python
import os, subprocess

# Check if log files exist
api_log = "logs/api.log" if os.path.exists("logs/api.log") else None
worker_log = "logs/worker.log" if os.path.exists("logs/worker.log") else None

# Check if docker-compose services are running
docker_available = subprocess.run(
    ["docker-compose", "ps", "--services", "--filter", "status=running"],
    capture_output=True, text=True
).returncode == 0
```

Print: `Tailing logs for tenant: <tenant_id>`
Print which source is being used.

### Step 2 — Stream and filter

**Option A — Log files:**
```python
import subprocess, json

cmd = ["tail", "-f", "-n", "100", api_log, worker_log]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)

for line in proc.stdout:
    line = line.strip()
    if not line:
        continue
    try:
        entry = json.loads(line)
        if entry.get("tenant_id") == tenant_id:
            pretty_print_log(entry)
    except json.JSONDecodeError:
        # Non-JSON line (startup messages, etc.) — show if it mentions tenant_id
        if tenant_id in line:
            print(f"  [raw] {line}")
```

**Option B — Docker compose:**
```bash
docker-compose logs -f --tail=100 api worker 2>&1 | \
  python -c "
import sys, json
tenant_id = sys.argv[1]
for line in sys.stdin:
    line = line.strip()
    try:
        entry = json.loads(line)
        if entry.get('tenant_id') == tenant_id:
            # pretty print
    except: pass
" <tenant_id>
```

### Step 3 — Pretty-print format

Color code by level:
- `DEBUG` → grey
- `INFO` → white
- `WARNING` → yellow
- `ERROR` → red
- `CRITICAL` → red + bold

Format:
```
[HH:MM:SS.mmm] [LEVEL ]  event_name          key=value  key=value
[14:32:01.123] [INFO  ]  query_received      tenant_id=acme-corp  question="What was revenue?"
[14:32:01.456] [INFO  ]  cache_miss          tenant_id=acme-corp  query_hash=abc123
[14:32:01.890] [INFO  ]  retrieval_start     tenant_id=acme-corp  query="What was revenue?"
[14:32:02.120] [INFO  ]  bm25_complete       tenant_id=acme-corp  hits=10  latency_ms=45
[14:32:02.430] [INFO  ]  dense_complete      tenant_id=acme-corp  hits=10  latency_ms=310
[14:32:02.445] [INFO  ]  reranker_complete   tenant_id=acme-corp  top_score=0.89  latency_ms=85
[14:32:02.446] [INFO  ]  threshold_pass      tenant_id=acme-corp  score=0.89  threshold=0.72
[14:32:03.890] [INFO  ]  query_complete      tenant_id=acme-corp  latency_ms=1890  cache_written=true
```

Strip the raw JSON from output — only show the formatted version.

### Step 4 — Special highlights

Print a separator line when a new query begins (detect `query_received` event):
```
────────────────────────────────────────────────────────────
[14:32:01.123] [INFO] query_received  "What was revenue?"
```

Print a summary line when a query completes (detect `query_complete` event):
```
  → Answer in 1890ms | chunks=5 | score=0.89 | cache=WRITTEN
```

Print an alert when errors occur:
```
!!! [14:32:05.001] [ERROR] ingestion_failed  doc_id=42  error="PDF parsing failed"
```

### Step 5 — Keyboard interrupt

Catch `KeyboardInterrupt` (Ctrl-C) gracefully:
```
^C
Stopped tailing. Last log line: <timestamp>
```

## Notes

- If no logs appear for 30 seconds: `(no activity for tenant <tenant_id> in 30s — is the API running?)`
- Use `--since 1h` flag equivalent for docker if user wants historical logs
- Logs with `tenant_id=null` or missing are shown in dim style with a note
