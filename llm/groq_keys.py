"""Round-robin Groq API key rotation — avoids per-key rate limits during RAGAS eval."""
from __future__ import annotations

from itertools import cycle

from config.settings import settings

_keys: list[str] = [k for k in [settings.groq_api_key, settings.groq_api_key_2] if k]
if not _keys:
    raise RuntimeError("No Groq API keys configured. Set GROQ_API_KEY in .env.")

_pool = cycle(_keys)


def next_groq_key() -> str:
    return next(_pool)


