"""Gemini backend — Google AI Studio OpenAI-compatible API."""
from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from config.logging import get_logger
from config.settings import settings

log = get_logger()

_GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


class GeminiBackend:
    def __init__(self, model: str | None = None) -> None:
        try:
            if not settings.gemini_api_key:
                raise ValueError("GEMINI_API_KEY not set")
            self.model = model or settings.gemini_model
            self._client = AsyncOpenAI(
                base_url=_GEMINI_BASE,
                api_key=settings.gemini_api_key,
            )
            log.info("gemini.init", model=self.model)
        except Exception as exc:
            log.error("gemini.init_failed", model=model, error=str(exc))
            raise
        finally:
            log.debug("gemini.init_exit")

    async def generate(self, prompt: str, system: str = "") -> str:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            resp = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
            )
            text = resp.choices[0].message.content or ""
            log.debug("gemini.generated", model=self.model, chars=len(text))
            return text
        except Exception as exc:
            log.error("gemini.generate_failed", model=self.model, error=str(exc))
            raise

    async def stream(self, prompt: str, system: str = "") -> AsyncIterator[str]:
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        try:
            async with await self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                stream=True,
            ) as resp:
                async for chunk in resp:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except Exception as exc:
            log.error("gemini.stream_failed", model=self.model, error=str(exc))
            raise
