from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class TogetherChat(LLMProvider):
    """
    Together AI provider for fast inference of open-source models.
    Supports Llama, Mistral, CodeLlama, RedPajama, and other models with competitive pricing.
    """
    
    def __init__(self, model: str = "meta-llama/Llama-2-7b-chat-hf", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Together provider.
        
        Args:
            model: Together model name (e.g., 'meta-llama/Llama-2-7b-chat-hf')
            api_key: Together API key (defaults to TOGETHER_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        
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
                "Together API key is required. Set TOGETHER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Validate model name
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        popular_models = [
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf", 
            "meta-llama/Llama-2-70b-chat-hf",
            "meta-llama/Llama-3-8b-chat-hf",
            "meta-llama/Llama-3-70b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "codellama/CodeLlama-7b-Instruct-hf",
            "codellama/CodeLlama-13b-Instruct-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "togethercomputer/RedPajama-INCITE-7B-Chat",
            "togethercomputer/RedPajama-INCITE-13B-Chat",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "teknium/OpenHermes-2.5-Mistral-7B",
            "microsoft/Orca-2-7b",
            "microsoft/Orca-2-13b"
        ]
        
        if self.model not in popular_models:
            print(f"Info: Using custom model '{self.model}'. Popular models: {popular_models[:5]}...")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Together model.
        
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
        
        # Convert messages to OpenAI-compatible format (Together follows OpenAI format)
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
        if "top_k" in merged_kwargs:
            payload["top_k"] = merged_kwargs["top_k"]
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "RAGtester/1.0"
        }
        
        try:
            # Make the API call to Together
            response = requests.post(
                "https://api.together.xyz/v1/chat/completions",
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
            raise RuntimeError(f"Together API call failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Together API error: {e}")
