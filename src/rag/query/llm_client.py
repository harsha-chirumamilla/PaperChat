import requests
import httpx
import json
import logging
import asyncio
from typing import AsyncIterator
from src.common.config import Config

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client for multiple API providers with async support."""
    
    def __init__(self, provider: str):
        """
        Initialize LLM client for specified provider.
        
        Args:
            provider: One of 'openai', 'anthropic', or 'gemini'
        """
        self.provider = provider.lower()
        self._async_client = None
        self._init_client()
    
    def _get_async_client(self) -> httpx.AsyncClient:
        """Lazy-load async HTTP client."""
        if self._async_client is None:
            self._async_client = httpx.AsyncClient(timeout=60.0)
        return self._async_client
    
    async def close(self):
        """Close async client resources."""
        if self._async_client:
            await self._async_client.aclose()
            self._async_client = None
    
    def _init_client(self):
        """Initialize and validate provider configuration."""
        if self.provider == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable must be set.")
            self.model_name = Config.OPENAI_MODEL_ID
            logger.info("OpenAI configured with model: %s", self.model_name)
            
        elif self.provider == "anthropic":
            if not Config.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY environment variable must be set.")
            self.model_name = Config.ANTHROPIC_MODEL_ID
            logger.info("Anthropic configured with model: %s", self.model_name)
            
        elif self.provider == "gemini":
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY environment variable must be set.")
            self.model_name = Config.GEMINI_MODEL_ID
            logger.info("Gemini configured with model: %s", self.model_name)
            
        else:
            raise ValueError(
                f"Unknown MODEL_PROVIDER: {self.provider}. "
                f"Supported providers: 'openai', 'anthropic', 'gemini'"
            )
    
    def generate(self, prompt: str) -> str:
        """
        Generate text response from LLM using direct HTTP calls (sync wrapper).
        
        Args:
            prompt: The text prompt to send to the LLM
            
        Returns:
            Generated text response
        """
        try:
            # Run async method in sync context
            return asyncio.run(self.generate_async(prompt))
        except Exception as e:
            logger.error("Error in generate: %s", e)
            return f"Error generating answer with {self.provider}: {str(e)}"
    
    async def generate_async(self, prompt: str) -> str:
        """
        Generate text response from LLM using async HTTP calls.
        
        Args:
            prompt: The text prompt to send to the LLM
            
        Returns:
            Generated text response
        """
        try:
            if self.provider == "openai":
                return await self._generate_openai_async(prompt)
            elif self.provider == "anthropic":
                return await self._generate_anthropic_async(prompt)
            elif self.provider == "gemini":
                return await self._generate_gemini_async(prompt)
                
        except httpx.HTTPStatusError as e:
            return f"HTTP error with {self.provider}: {e.response.status_code}"
        except httpx.RequestError as e:
            return f"Request error with {self.provider}: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing {self.provider} response: {str(e)}"
        except Exception as e:
            return f"Unexpected error with {self.provider}: {str(e)}"
    
    async def _generate_openai_async(self, prompt: str) -> str:
        """Generate response using OpenAI API (async)."""
        client = self._get_async_client()
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": Config.TEMPERATURE,
                "max_tokens": Config.MAX_NEW_TOKENS
            }
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    
    async def _generate_anthropic_async(self, prompt: str) -> str:
        """Generate response using Anthropic API (async)."""
        client = self._get_async_client()
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": Config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "max_tokens": Config.MAX_NEW_TOKENS,
                "temperature": Config.TEMPERATURE,
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"].strip()
    
    async def _generate_gemini_async(self, prompt: str) -> str:
        """Generate response using Google Gemini API (async)."""
        client = self._get_async_client()
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": Config.GEMINI_API_KEY
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}]
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    
    async def stream_async(self, prompt: str) -> AsyncIterator[str]:
        """
        Stream response tokens from LLM (async generator).
        
        Args:
            prompt: The text prompt to send to the LLM
            
        Yields:
            Text chunks as they're generated
        """
        if self.provider == "openai":
            async for chunk in self._stream_openai(prompt):
                yield chunk
        elif self.provider == "anthropic":
            async for chunk in self._stream_anthropic(prompt):
                yield chunk
        elif self.provider == "gemini":
            async for chunk in self._stream_gemini(prompt):
                yield chunk
    
    async def _stream_openai(self, prompt: str) -> AsyncIterator[str]:
        """Stream response from OpenAI API."""
        client = self._get_async_client()
        async with client.stream(
            "POST",
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": Config.TEMPERATURE,
                "max_tokens": Config.MAX_NEW_TOKENS,
                "stream": True
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if delta := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                            yield delta
                    except json.JSONDecodeError:
                        continue
    
    async def _stream_anthropic(self, prompt: str) -> AsyncIterator[str]:
        """Stream response from Anthropic API."""
        client = self._get_async_client()
        async with client.stream(
            "POST",
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": Config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "max_tokens": Config.MAX_NEW_TOKENS,
                "temperature": Config.TEMPERATURE,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    try:
                        data = json.loads(data_str)
                        if data.get("type") == "content_block_delta":
                            if delta := data.get("delta", {}).get("text"):
                                yield delta
                    except json.JSONDecodeError:
                        continue
    
    async def _stream_gemini(self, prompt: str) -> AsyncIterator[str]:
        """Stream response from Gemini API."""
        client = self._get_async_client()
        async with client.stream(
            "POST",
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:streamGenerateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": Config.GEMINI_API_KEY
            },
            json={
                "contents": [{"parts": [{"text": prompt}]}]
            }
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    try:
                        data = json.loads(line)
                        if candidates := data.get("candidates", []):
                            if text := candidates[0].get("content", {}).get("parts", [{}])[0].get("text"):
                                yield text
                    except json.JSONDecodeError:
                        continue
    
    # Keep sync methods for backward compatibility
    def _generate_openai(self, prompt: str) -> str:
        """Generate response using OpenAI API (sync - deprecated)."""
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": Config.TEMPERATURE,
                "max_tokens": Config.MAX_NEW_TOKENS
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    
    def _generate_anthropic(self, prompt: str) -> str:
        """Generate response using Anthropic API."""
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": Config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": self.model_name,
                "max_tokens": Config.MAX_NEW_TOKENS,
                "temperature": Config.TEMPERATURE,
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["content"][0]["text"].strip()
    
    def _generate_gemini(self, prompt: str) -> str:
        """Generate response using Google Gemini API."""
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent",
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": Config.GEMINI_API_KEY
            },
            data=json.dumps({
                "contents": [{"parts": [{"text": prompt}]}]
            }),
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
