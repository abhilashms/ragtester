from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class GrokChat(LLMProvider):
    """
    Grok (xAI) LLM provider using the Grok API.
    Supports Grok-1, Grok-2, and other xAI models.
    """
    
    def __init__(self, model: str = "grok-beta", api_key: str = None, base_url: str = None, **kwargs: Any) -> None:
        """
        Initialize Grok provider.
        
        Args:
            model: Grok model name (e.g., 'grok-beta', 'grok-2')
            api_key: xAI API key (defaults to XAI_API_KEY environment variable)
            base_url: Custom API base URL (optional)
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.base_url = base_url or "https://api.x.ai/v1"
        
        # Store chat parameters separately from client parameters
        self.chat_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1024),
            'top_p': kwargs.get('top_p', 1.0)
        }
        
        # Store other parameters that might be used by the client
        self.client_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['temperature', 'max_tokens', 'top_p']}
        
        if not self.api_key:
            raise ValueError(
                "xAI API key is required. Set XAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Validate model name
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        valid_models = [
            # Grok-2 Series (Latest)
            "grok-2",
            "grok-2-1212",
            "grok-2-1212-beta",
            
            # Grok-1 Series
            "grok-beta",
            "grok-1",
            "grok-1-beta",
            
            # Future models (when available)
            "grok-3",
            "grok-3-beta",
        ]
        
        if self.model not in valid_models:
            self.logger.warning(f"Warning: Model '{self.model}' may not be available. Valid models: {valid_models}")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Grok model.
        
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

        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
        # Convert messages to OpenAI-compatible format (Grok API follows OpenAI format)
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
            "max_tokens": merged_chat_params.get("max_tokens", 1024),
            "temperature": merged_chat_params.get("temperature", 0.7),
            "stream": False
        }
        
        # Add optional parameters
        if "top_p" in merged_chat_params:
            payload["top_p"] = merged_chat_params["top_p"]
        if "frequency_penalty" in merged_chat_params:
            payload["frequency_penalty"] = merged_chat_params["frequency_penalty"]
        if "presence_penalty" in merged_chat_params:
            payload["presence_penalty"] = merged_chat_params["presence_penalty"]
        if "stop" in merged_chat_params:
            payload["stop"] = merged_chat_params["stop"]
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RAGtester/1.0"
        }
        
        try:
            # Make the API call
            response = requests.post(
                f"{self.base_url}/chat/completions",
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
            raise RuntimeError(f"Grok API call failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Grok API error: {e}")
