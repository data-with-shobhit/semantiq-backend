"""Query rewriting: HyDE for dense retrieval.

SPLADE (naver/splade-cocondenser-ensembledistil) handles vocabulary
expansion implicitly for sparse retrieval — no manual synonym dicts needed.
Works out of the box for financial, medical, legal, clinical, and scientific.

OOV caveat (technical domain only): code tokens may be under-weighted
by SPLADE. HyDE still helps here by generating natural-language
descriptions that bridge code ↔ query vocabulary.
"""
from __future__ import annotations

from config.logging import get_logger
from config.settings import settings

log = get_logger()

_HYDE_PROMPT = (
    "Write a short factual passage (2-3 sentences) that directly answers "
    "this question. Do not say you cannot answer. Just write the passage:\n\n"
)


def _get_hyde_llm():
    """Use Gemini for HyDE if key is set — saves Groq TPD tokens."""
    if settings.gemini_api_key:
        from llm.gemini_backend import GeminiBackend
        return GeminiBackend()
    return None


async def hyde_rewrite(query: str, llm_backend) -> str:
    """Generate a hypothetical document that answers the query.

    Tries Gemini first (cheaper). On any error (rate limit, quota) falls back
    to the main LLM backend so HyDE still runs — not degraded to raw query.
    """
    prompt = _HYDE_PROMPT + query
    system = "You are a helpful assistant."

    hyde_llm = _get_hyde_llm()
    if hyde_llm:
        try:
            hypothesis = await hyde_llm.generate(prompt, system=system)
            log.debug("rewriter.hyde", backend="gemini", original=query[:60], hypothesis=hypothesis[:80])
            return hypothesis
        except Exception as exc:
            log.warning("rewriter.hyde_gemini_fallback", error=str(exc))

    try:
        hypothesis = await llm_backend.generate(prompt, system=system)
        log.debug("rewriter.hyde", backend="main", original=query[:60], hypothesis=hypothesis[:80])
        return hypothesis
    except Exception as exc:
        log.warning("rewriter.hyde_failed", error=str(exc))
        return query
