from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class FireworksChat(LLMProvider):
    """
    Fireworks AI provider for fast inference of open-source models.
    Supports Llama, Mistral, CodeLlama, and other models with serverless inference.
    """
    
    def __init__(self, model: str = "accounts/fireworks/models/llama-v2-7b-chat", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Fireworks provider.
        
        Args:
            model: Fireworks model name (e.g., 'accounts/fireworks/models/llama-v2-7b-chat')
            api_key: Fireworks API key (defaults to FIREWORKS_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("FIREWORKS_API_KEY")
        self.kwargs = kwargs
        
        if not self.api_key:
            raise ValueError(
                "Fireworks API key is required. Set FIREWORKS_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Validate model name
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        popular_models = [
            "accounts/fireworks/models/llama-v2-7b-chat",
            "accounts/fireworks/models/llama-v2-13b-chat", 
            "accounts/fireworks/models/llama-v2-70b-chat",
            "accounts/fireworks/models/llama-v3-8b-instruct",
            "accounts/fireworks/models/llama-v3-70b-instruct",
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            "accounts/fireworks/models/mistral-7b-instruct",
            "accounts/fireworks/models/codellama-7b-instruct",
            "accounts/fireworks/models/codellama-13b-instruct",
            "accounts/fireworks/models/codellama-34b-instruct",
            "accounts/fireworks/models/yi-6b-200k",
            "accounts/fireworks/models/yi-34b-200k"
        ]
        
        if self.model not in popular_models:
            print(f"Info: Using custom model '{self.model}'. Popular models: {popular_models[:5]}...")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Fireworks model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests package is required. Install with: pip install requests"
            )

        # Merge kwargs
        merged_kwargs = {**self.kwargs, **kwargs}
        
        # Convert messages to OpenAI-compatible format (Fireworks follows OpenAI format)
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": openai_messages,
            "max_tokens": merged_kwargs.get("max_tokens", 1024),
            "temperature": merged_kwargs.get("temperature", 0.7),
            "stream": False
        }
        
        # Add optional parameters
        if "top_p" in merged_kwargs:
            payload["top_p"] = merged_kwargs["top_p"]
        if "frequency_penalty" in merged_kwargs:
            payload["frequency_penalty"] = merged_kwargs["frequency_penalty"]
        if "presence_penalty" in merged_kwargs:
            payload["presence_penalty"] = merged_kwargs["presence_penalty"]
        if "stop" in merged_kwargs:
            payload["stop"] = merged_kwargs["stop"]
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RAGtester/1.0"
        }
        
        try:
            # Make the API call to Fireworks
            response = requests.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=60
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract the response content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                return ""
                
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Fireworks API call failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Fireworks API error: {e}")
