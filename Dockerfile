FROM python:3.11-slim

WORKDIR /app

# System deps for unstructured, pypdf, tokenizers, torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download SPLADE model at build time — avoids 60s cold start
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForMaskedLM; \
AutoTokenizer.from_pretrained('naver/splade-v3'); \
AutoModelForMaskedLM.from_pretrained('naver/splade-v3')"

COPY . .

# Default: API server. Override for worker:
# --command celery --args "-A,ingestion.tasks,worker,..."
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
