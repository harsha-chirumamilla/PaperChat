import requests
import json
from src.common.config import Config


class LLMClient:
    """Unified LLM client for multiple API providers using direct HTTP calls."""
    
    def __init__(self, provider: str):
        """
        Initialize LLM client for specified provider.
        
        Args:
            provider: One of 'openai', 'anthropic', or 'gemini'
        """
        self.provider = provider.lower()
        self._init_client()
    
    def _init_client(self):
        """Initialize and validate provider configuration."""
        if self.provider == "openai":
            if not Config.OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY environment variable must be set.")
            self.model_name = Config.OPENAI_MODEL_ID
            print(f"OpenAI configured with model: {self.model_name}")
            
        elif self.provider == "anthropic":
            if not Config.ANTHROPIC_API_KEY:
                raise ValueError("ANTHROPIC_API_KEY environment variable must be set.")
            self.model_name = Config.ANTHROPIC_MODEL_ID
            print(f"Anthropic configured with model: {self.model_name}")
            
        elif self.provider == "gemini":
            if not Config.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY environment variable must be set.")
            self.model_name = Config.GEMINI_MODEL_ID
            print(f"Gemini configured with model: {self.model_name}")
            
        else:
            raise ValueError(
                f"Unknown MODEL_PROVIDER: {self.provider}. "
                f"Supported providers: 'openai', 'anthropic', 'gemini'"
            )
    
    def generate(self, prompt: str) -> str:
        """
        Generate text response from LLM using direct HTTP calls.
        
        Args:
            prompt: The text prompt to send to the LLM
            
        Returns:
            Generated text response
        """
        try:
            if self.provider == "openai":
                return self._generate_openai(prompt)
            elif self.provider == "anthropic":
                return self._generate_anthropic(prompt)
            elif self.provider == "gemini":
                return self._generate_gemini(prompt)
                
        except requests.exceptions.RequestException as e:
            return f"Error generating answer with {self.provider}: {str(e)}"
        except (KeyError, IndexError) as e:
            return f"Error parsing {self.provider} response: {str(e)}"
        except Exception as e:
            return f"Unexpected error with {self.provider}: {str(e)}"
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate response using OpenAI API."""
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
