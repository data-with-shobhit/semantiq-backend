"""OpenRouter backend — OpenAI-compatible API with free model access."""
from __future__ import annotations

from typing import AsyncIterator

from openai import AsyncOpenAI

from config.logging import get_logger
from config.settings import settings

log = get_logger()

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


class OpenRouterBackend:
    def __init__(self, model: str | None = None) -> None:
        try:
            if not settings.openrouter_api_key:
                raise ValueError("OPENROUTER_API_KEY not set")
            self.model = model or settings.openrouter_model
            self._client = AsyncOpenAI(
                base_url=_OPENROUTER_BASE,
                api_key=settings.openrouter_api_key,
            )
            log.info("openrouter.init", model=self.model)
        except Exception as exc:
            log.error("openrouter.init_failed", model=model, error=str(exc))
            raise
        finally:
            log.debug("openrouter.init_exit")

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
            log.debug("openrouter.generated", model=self.model, chars=len(text))
            return text
        except Exception as exc:
            log.error("openrouter.generate_failed", model=self.model, error=str(exc))
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
            log.error("openrouter.stream_failed", model=self.model, error=str(exc))
            raise
