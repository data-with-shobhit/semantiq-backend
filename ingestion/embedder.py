"""Domain-aware embedder — Voyage AI API (preferred) → Jina AI → local SentenceTransformer."""
from __future__ import annotations

from functools import lru_cache

from config.logging import get_logger
from config.settings import settings
from ingestion.model_registry import get_model_spec

log = get_logger()

_VOYAGE_BASE = "https://api.voyageai.com/v1/embeddings"
_JINA_BASE   = "https://api.jina.ai/v1/embeddings"


# ── Voyage API ────────────────────────────────────────────────────────────────

def _voyage_encode(texts: list[str], model: str, input_type: str) -> list[list[float]]:
    if not texts:
        return []
    import time

    import httpx

    # Sanitise: replace empty strings with a placeholder so API doesn't reject
    clean = [t if t and t.strip() else "." for t in texts]
    for attempt in range(5):
        resp = httpx.post(
            _VOYAGE_BASE,
            headers={"Authorization": f"Bearer {settings.voyage_api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": clean, "input_type": input_type},
            timeout=120,
        )
        if resp.status_code == 429:
            wait = 2 ** attempt * 10
            log.warning("embedder.voyage_rate_limit", attempt=attempt, wait_s=wait)
            time.sleep(wait)
            continue
        if resp.status_code == 400:
            log.error("embedder.voyage_400", model=model, n=len(clean), sample=clean[0][:80] if clean else "")
        resp.raise_for_status()
        data = resp.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
    resp.raise_for_status()


# ── Jina API ──────────────────────────────────────────────────────────────────

def _jina_encode(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    import httpx
    resp = httpx.post(
        _JINA_BASE,
        headers={"Authorization": f"Bearer {settings.jina_api_key}", "Content-Type": "application/json"},
        json={"model": settings.jina_model, "input": texts, "task": task},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


# ── Local SentenceTransformer ─────────────────────────────────────────────────

@lru_cache(maxsize=3)
def _load_model(hf_id: str):
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("embedder.loading", hf_id=hf_id, device=device)
    model = SentenceTransformer(hf_id, device=device)
    log.info("embedder.loaded", hf_id=hf_id)
    return model


# ── Embedder ──────────────────────────────────────────────────────────────────

class Embedder:
    def __init__(self, domain: str) -> None:
        try:
            spec = get_model_spec(domain)
            self.domain = domain
            self.model_id = spec["model_id"]
            self.dim = spec["dim"]
            self.backend = spec["backend"]

            if self.backend == "voyage" and settings.voyage_api_key:
                self._mode = "voyage"
                log.info("embedder.voyage", model=self.model_id, domain=domain)
            elif settings.jina_api_key:
                self._mode = "jina"
                log.info("embedder.jina", model=settings.jina_model, domain=domain)
            else:
                self._mode = "local"
                self._model = _load_model(self.model_id)
                log.info("embedder.local", model=self.model_id, domain=domain)
        except Exception as exc:
            log.error("embedder.init_failed", domain=domain, error=str(exc))
            raise
        finally:
            log.debug("embedder.init_exit", domain=domain)

    def encode(self, texts: list[str], batch_size: int = 64, normalize: bool = True) -> list[list[float]]:
        if not texts:
            return []
        try:
            if self._mode == "voyage":
                import time
                results = []
                for i in range(0, len(texts), batch_size):
                    results.extend(_voyage_encode(texts[i:i + batch_size], self.model_id, "document"))
                    if i + batch_size < len(texts):
                        time.sleep(0.5)
                log.debug("embedder.encoded", domain=self.domain, n=len(texts), backend="voyage")
                return results
            if self._mode == "jina":
                results = []
                for i in range(0, len(texts), batch_size):
                    results.extend(_jina_encode(texts[i:i + batch_size], task="retrieval.passage"))
                log.debug("embedder.encoded", domain=self.domain, n=len(texts), backend="jina")
                return results
            vectors = self._model.encode(
                texts, batch_size=32, normalize_embeddings=normalize, show_progress_bar=False,
            )
            log.debug("embedder.encoded", domain=self.domain, n=len(texts), backend="local")
            return vectors.tolist()
        except Exception as exc:
            log.error("embedder.encode_failed", domain=self.domain, error=str(exc))
            raise

    def encode_one(self, text: str) -> list[float]:
        if self._mode == "voyage":
            return _voyage_encode([text], self.model_id, "query")[0]
        if self._mode == "jina":
            return _jina_encode([text], task="retrieval.query")[0]
        return self.encode([text])[0]
