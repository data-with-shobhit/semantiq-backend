"""Groq backend — OpenAI-compatible cloud API."""
from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from config.logging import get_logger
from config.settings import settings

log = get_logger()

_GROQ_BASE = "https://api.groq.com/openai/v1"


class GroqBackend:
    def __init__(self, model: str | None = None) -> None:
        try:
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY not set")
            self.model = model or settings.groq_generation_model
            self._client = AsyncOpenAI(
                base_url=_GROQ_BASE,
                api_key=settings.groq_api_key,
            )
            log.info("groq.init", model=self.model)
        except Exception as exc:
            log.error("groq.init_failed", model=model, error=str(exc))
            raise
        finally:
            log.debug("groq.init_exit")

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
                timeout=60,
            )
            text = resp.choices[0].message.content or ""
            log.debug("groq.generated", model=self.model, chars=len(text))
            return text
        except Exception as exc:
            log.error("groq.generate_failed", model=self.model, error=str(exc), prompt_chars=sum(len(m["content"]) for m in messages))
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
            log.error("groq.stream_failed", model=self.model, error=str(exc))
            raise
