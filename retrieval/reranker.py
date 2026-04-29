"""Voyage AI rerank-2.5 reranker + score threshold gate."""
from __future__ import annotations

import math
from functools import lru_cache

import httpx

from config.logging import get_logger
from config.settings import settings

log = get_logger()

_THRESHOLD = settings.score_threshold
_TOP_N = settings.max_chunks
_VOYAGE_RERANK_URL = "https://api.voyageai.com/v1/rerank"


def _voyage_rerank(query: str, documents: list[str], top_k: int) -> list[dict]:
    resp = httpx.post(
        _VOYAGE_RERANK_URL,
        headers={"Authorization": f"Bearer {settings.voyage_api_key}", "Content-Type": "application/json"},
        json={"model": settings.voyage_reranker_model, "query": query, "documents": documents, "top_k": top_k},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["data"]


def rerank(query: str, candidates: list[dict], threshold: float = _THRESHOLD) -> list[dict]:
    if not candidates:
        return []

    try:
        if settings.voyage_api_key:
            docs = [c["text"] for c in candidates]
            ranked = _voyage_rerank(query, docs, top_k=_TOP_N)
            results = []
            for item in ranked:
                c = candidates[item["index"]]
                score = item["relevance_score"]
                results.append({**c, "rerank_score": round(score, 4)})
            log.info("reranker.voyage_done", kept=len(results), top_score=results[0]["rerank_score"] if results else 0)
        else:
            results = _local_rerank(query, candidates)

        top_score = results[0]["rerank_score"] if results else 0.0
        if top_score < threshold:
            log.info("reranker.below_threshold_llm_decides", top_score=top_score)
        else:
            log.info("reranker.above_threshold", top_score=top_score)

        return results

    except Exception as exc:
        log.warning("reranker.failed_returning_candidates", error=str(exc))
        return candidates[:_TOP_N]


def _local_rerank(query: str, candidates: list[dict]) -> list[dict]:
    """Fallback: CrossEncoder when Voyage API key not set."""
    from sentence_transformers import CrossEncoder

    @lru_cache(maxsize=1)
    def _load():
        log.info("reranker.loading_local", model=settings.reranker_model)
        return CrossEncoder(settings.reranker_model)

    model = _load()
    pairs = [(query, c["text"]) for c in candidates]
    raw_scores = model.predict(pairs).tolist()
    scores = [1.0 / (1.0 + math.exp(-s)) for s in raw_scores]
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [{**c, "rerank_score": round(s, 4)} for c, s in ranked[:_TOP_N]]
