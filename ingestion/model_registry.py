"""Single source of truth for domain → embedding model mapping.

Never hardcode model IDs or dims anywhere else — always import from here.
"""
from __future__ import annotations

from typing import TypedDict

from config.logging import get_logger

log = get_logger()


class ModelSpec(TypedDict):
    model_id: str   # Voyage model name or HF ID
    dim: int
    backend: str    # "voyage" | "local"


_VOYAGE_4     = {"model_id": "voyage-4",          "dim": 1024, "backend": "voyage"}
_VOYAGE_FIN   = {"model_id": "voyage-finance-2", "dim": 1024, "backend": "voyage"}
_VOYAGE_LAW   = {"model_id": "voyage-law-2",     "dim": 1024, "backend": "voyage"}
_VOYAGE_CODE  = {"model_id": "voyage-code-3",    "dim": 1024, "backend": "voyage"}

MODEL_REGISTRY: dict[str, ModelSpec] = {
    "general":    _VOYAGE_4,
    "financial":  _VOYAGE_FIN,
    "medical":    _VOYAGE_4,
    "clinical":   _VOYAGE_4,
    "legal":      _VOYAGE_LAW,
    "scientific": _VOYAGE_4,
    "technical":  _VOYAGE_CODE,
}

DOMAINS = frozenset(MODEL_REGISTRY.keys())


def get_model_spec(domain: str) -> ModelSpec:
    try:
        if domain not in MODEL_REGISTRY:
            raise ValueError(f"Unknown domain '{domain}'. Valid: {sorted(DOMAINS)}")
        return MODEL_REGISTRY[domain]
    except ValueError:
        raise
    except Exception as exc:
        log.error("model_registry.get_spec_failed", domain=domain, error=str(exc))
        raise
    finally:
        log.debug("model_registry.get_spec_exit", domain=domain)
