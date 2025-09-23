from __future__ import annotations

import json
import requests
from typing import Any, Sequence
from urllib.parse import urljoin, urlparse

from .base import LLMProvider
from ..types import LLMMessage


class OllamaLLM(LLMProvider):
    """
    Ollama LLM provider for local Ollama server instances.
    Supports both local and remote Ollama servers.
    """
    
    def __init__(self, model: str = None, base_url: str = None, **kwargs: Any) -> None:
        """
        Initialize Ollama LLM provider.
        
        Args:
            model: Model name (e.g., "llama2", "codellama", "mistral")
            base_url: Base URL of the Ollama server (e.g., "http://localhost:11434")
            **kwargs: Additional parameters
        """
        self.model = model
        self.base_url = base_url or "http://localhost:11434"
        
        # Ensure base_url doesn't end with /
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]
        
        # Default parameters
        self.default_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 512),
            'top_p': kwargs.get('top_p', 0.9),
        }
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama server at {self.base_url}: {e}")
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using the Ollama model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        # Convert messages to Ollama format
        ollama_messages = []
        for message in messages:
            ollama_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        # Merge generation parameters
        gen_params = {**self.default_params, **kwargs}
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": gen_params.get('temperature', 0.7),
                "num_predict": gen_params.get('max_tokens', 512),
                "top_p": gen_params.get('top_p', 0.9),
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API request failed: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Invalid response from Ollama server: {e}")
    
    def _format_messages(self, messages: Sequence[LLMMessage]) -> str:
        """Convert message sequence to prompt format (not used in Ollama)."""
        # Ollama handles message formatting internally
        return ""
