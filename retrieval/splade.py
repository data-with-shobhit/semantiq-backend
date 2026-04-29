"""SPLADE sparse encoder with BM25 fallback.

Primary:  prithivida/Splade_PP_en_v1 — learned sparse retrieval, implicit vocabulary expansion.
          Works out of the box for financial, medical, legal, clinical, scientific.
Fallback: rank-bm25 — exact match, always available, handles OOV cases
          (error codes, internal IDs, code tokens) where SPLADE struggles.

Pattern: try SPLADE → if it fails or returns empty → fall back to BM25.

OOV caveat (technical domain only): SPLADE trained on general/web text;
code-specific tokens may be under-weighted. BM25 fallback covers these cases.

Note: `from transformers import ...` is intentionally inside _load_splade()
to keep module-level import cost near zero — the model is ~400MB and only
needed when encoding is actually called.
"""
from __future__ import annotations

from functools import lru_cache

import torch
from rank_bm25 import BM25Okapi

from config.logging import get_logger

log = get_logger()

_SPLADE_MODEL = "prithivida/Splade_PP_en_v1"  # public, no gating
_MAX_LENGTH = 512


@lru_cache(maxsize=1)
def _load_splade():
    # Deferred import: transformers is heavy; only load on first encode call.
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    log.info("splade.loading", model=_SPLADE_MODEL)
    try:
        tokenizer = AutoTokenizer.from_pretrained(_SPLADE_MODEL)
        model = AutoModelForMaskedLM.from_pretrained(_SPLADE_MODEL)
        model.eval()
        log.info("splade.loaded")
        return tokenizer, model
    except Exception as exc:
        log.error("splade.load_failed", model=_SPLADE_MODEL, error=str(exc))
        raise


def encode(text: str) -> tuple[list[int], list[float]]:
    """Encode text → (indices, values) sparse vector. Returns empty on failure."""
    try:
        return _encode_splade(text)
    except Exception as exc:
        log.warning("splade.encode_failed", error=str(exc))
        return [], []


def _encode_splade(text: str) -> tuple[list[int], list[float]]:
    try:
        tokenizer, model = _load_splade()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=_MAX_LENGTH,
            padding=False,
        )
        with torch.no_grad():
            output = model(**inputs)

        vec = torch.log1p(torch.relu(output.logits)).max(dim=1).values.squeeze(0)
        nonzero_mask = vec > 0
        indices = nonzero_mask.nonzero(as_tuple=False).squeeze(1).tolist()
        values = vec[nonzero_mask].tolist()

        log.debug("splade.encoded", nnz=len(indices))
        return indices, values
    except Exception as exc:
        log.error("splade.encode_splade_failed", error=str(exc))
        raise
    finally:
        log.debug("splade.encode_splade_exit")


def encode_batch(texts: list[str]) -> list[tuple[list[int], list[float]]]:
    try:
        return [encode(t) for t in texts]
    except Exception as exc:
        log.error("splade.encode_batch_failed", error=str(exc))
        return [([], [])] * len(texts)
    finally:
        log.debug("splade.encode_batch_exit", n=len(texts))


def bm25_search(query: str, candidates: list[dict]) -> list[dict]:
    """BM25 fallback over an in-memory candidate list."""
    if not candidates:
        return []
    try:
        tokenized = [c["text"].lower().split() for c in candidates]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        log.debug("splade.bm25_fallback", candidates=len(candidates))
        return [c for c, _ in ranked]
    except Exception as exc:
        log.warning("bm25.search_failed", error=str(exc))
        return candidates
