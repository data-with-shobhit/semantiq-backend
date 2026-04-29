"""LLMBackend protocol — swap LM Studio ↔ Anthropic via env var."""
from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    async def generate(self, prompt: str, system: str = "") -> str:
        """Single-turn generation."""
        ...

    async def stream(self, prompt: str, system: str = "") -> AsyncIterator[str]:
        """Streaming generation — yields token chunks."""
        ...
