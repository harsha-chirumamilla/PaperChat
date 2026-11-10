"""Small HTTP client for supported text generation providers."""

import asyncio
import logging

import httpx

from src.common.config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, provider: str):
        self.provider = provider.lower()
        self._async_client = None
        settings = {
            "openai": (Config.OPENAI_API_KEY, Config.OPENAI_MODEL_ID),
            "anthropic": (Config.ANTHROPIC_API_KEY, Config.ANTHROPIC_MODEL_ID),
            "gemini": (Config.GEMINI_API_KEY, Config.GEMINI_MODEL_ID),
        }
        if self.provider not in settings:
            raise ValueError(f"Unsupported MODEL_PROVIDER: {self.provider}")
        self.api_key, self.model_name = settings[self.provider]
        if not self.api_key:
            raise ValueError(f"API key for {self.provider} must be configured")

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=60.0)
        return self._async_client

    async def close(self):
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None

    def generate(self, prompt: str) -> str:
        """Synchronous entry point for Celery workers."""
        async def run_and_close():
            try:
                return await self.generate_async(prompt)
            finally:
                await self.close()

        try:
            return asyncio.run(run_and_close())
        except Exception:
            logger.exception("LLM generation failed for provider %s", self.provider)
            raise

    async def generate_async(self, prompt: str) -> str:
        client = self._get_async_client()
        if self.provider == "openai":
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": Config.TEMPERATURE,
                    "max_tokens": Config.MAX_NEW_TOKENS,
                },
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

        if self.provider == "anthropic":
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                json={
                    "model": self.model_name,
                    "max_tokens": Config.MAX_NEW_TOKENS,
                    "temperature": Config.TEMPERATURE,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            response.raise_for_status()
            return response.json()["content"][0]["text"].strip()

        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": Config.TEMPERATURE,
                    "maxOutputTokens": Config.MAX_NEW_TOKENS,
                },
            },
        )
        response.raise_for_status()
        return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
