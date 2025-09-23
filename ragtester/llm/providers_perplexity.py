from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class PerplexityChat(LLMProvider):
    """
    Perplexity AI provider with real-time web search capabilities.
    Supports Perplexity models with search-augmented responses for factual accuracy.
    """
    
    def __init__(self, model: str = "llama-3.1-sonar-small-128k-online", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Perplexity provider.
        
        Args:
            model: Perplexity model name (e.g., 'llama-3.1-sonar-small-128k-online')
            api_key: Perplexity API key (defaults to PERPLEXITY_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("PERPLEXITY_API_KEY")
        self.kwargs = kwargs
        
        if not self.api_key:
            raise ValueError(
                "Perplexity API key is required. Set PERPLEXITY_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Validate model name
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        valid_models = [
            "llama-3.1-sonar-small-128k-online",
            "llama-3.1-sonar-small-128k-chat",
            "llama-3.1-sonar-medium-128k-online", 
            "llama-3.1-sonar-medium-128k-chat",
            "llama-3.1-sonar-large-128k-online",
            "llama-3.1-sonar-large-128k-chat",
            "llama-3.1-sonar-huge-128k-online",
            "llama-3.1-sonar-huge-128k-chat",
            "llama-3.1-8b-instruct",
            "llama-3.1-70b-instruct",
            "llama-3.1-405b-instruct",
            "mixtral-8x7b-instruct",
            "mixtral-8x22b-instruct",
            "nous-hermes-2-mixtral-8x7b-dpo",
            "deepseek-coder-6.7b-instruct",
            "deepseek-coder-33b-instruct"
        ]
        
        if self.model not in valid_models:
            print(f"Warning: Model '{self.model}' may not be available. Valid models: {valid_models[:5]}...")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Perplexity model with optional web search.
        
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
        
        # Convert messages to Perplexity format
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
            "temperature": merged_kwargs.get("temperature", 0.2),  # Lower default for factual accuracy
            "stream": False
        }
        
        # Add optional parameters
        if "top_p" in merged_kwargs:
            payload["top_p"] = merged_kwargs["top_p"]
        if "top_k" in merged_kwargs:
            payload["top_k"] = merged_kwargs["top_k"]
        if "stop" in merged_kwargs:
            payload["stop"] = merged_kwargs["stop"]
        
        # Add search parameter for online models
        if "online" in self.model:
            payload["search_domain_filter"] = merged_kwargs.get("search_domain_filter", ["perplexity.ai"])
            payload["search_recency_filter"] = merged_kwargs.get("search_recency_filter", "month")
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RAGtester/1.0"
        }
        
        try:
            # Make the API call to Perplexity
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
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
            raise RuntimeError(f"Perplexity API call failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Perplexity API error: {e}")
