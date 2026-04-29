"""Anthropic Claude backend — optional, activated via LLM_PROVIDER=anthropic."""
from __future__ import annotations

from typing import AsyncIterator

from config.logging import get_logger
from config.settings import settings

try:
    import anthropic as _anthropic_lib
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


log = get_logger()

_MODEL = "claude-sonnet-4-6"


class AnthropicBackend:
    def __init__(self, model: str = _MODEL) -> None:
        try:
            if not _ANTHROPIC_AVAILABLE:
                raise RuntimeError("Install anthropic: uv add anthropic")
            self._client = _anthropic_lib.AsyncAnthropic(api_key=settings.anthropic_api_key)
            self.model = model
        except Exception as exc:
            log.error("anthropic.init_failed", model=model, error=str(exc))
            raise
        finally:
            log.debug("anthropic.init_exit", model=model)

    async def generate(self, prompt: str, system: str = "") -> str:
        try:
            resp = await self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=system or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            log.debug("llm.generated", model=self.model, chars=len(text))
            return text
        except Exception as exc:
            log.error("anthropic.generate_failed", model=self.model, error=str(exc))
            raise

    async def stream(self, prompt: str, system: str = "") -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(
                model=self.model,
                max_tokens=2048,
                system=system or "You are a helpful assistant.",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as exc:
            log.error("anthropic.stream_failed", model=self.model, error=str(exc))
            raise


def get_llm_backend():
    """Factory — reads LLM_PROVIDER from settings."""
    try:
        if settings.llm_provider == "anthropic":
            return AnthropicBackend()
        if settings.llm_provider == "groq":
            from llm.groq_backend import GroqBackend
            return GroqBackend()
        if settings.llm_provider == "openrouter":
            from llm.openrouter_backend import OpenRouterBackend
            return OpenRouterBackend()
        if settings.llm_provider == "gemini":
            from llm.gemini_backend import GeminiBackend
            return GeminiBackend()
        raise ValueError(f"Unknown LLM provider: '{settings.llm_provider}'. Valid: anthropic, groq, openrouter, gemini")
    except Exception as exc:
        log.error("llm.get_backend_failed", provider=settings.llm_provider, error=str(exc))
        raise
    finally:
        log.debug("llm.get_backend_exit", provider=settings.llm_provider)
