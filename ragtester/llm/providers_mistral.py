from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class MistralChat(LLMProvider):
    """
    Mistral AI LLM provider using the Mistral API.
    Supports Mistral Large, Mistral Medium, Mistral Small, and other Mistral models.
    """
    
    def __init__(self, model: str = "mistral-large-latest", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Mistral provider.
        
        Args:
            model: Mistral model name (e.g., 'mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest')
            api_key: Mistral API key (defaults to MISTRAL_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        
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
                "Mistral API key is required. Set MISTRAL_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Validate model name
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        valid_models = [
            "mistral-large-latest",
            "mistral-medium-latest", 
            "mistral-small-latest",
            "mistral-7b-instruct",
            "mistral-8x7b-instruct",
            "mistral-8x22b-instruct",
            "codestral-latest",
            "pixtral-12b-2409",
            "pixtral-latest"
        ]
        
        if self.model not in valid_models:
            print(f"Warning: Model '{self.model}' may not be available. Valid models: {valid_models}")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Mistral model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            from mistralai.client import MistralClient
            from mistralai.models.chat_completion import ChatMessage
        except ImportError:
            raise ImportError(
                "mistralai package is required. Install with: pip install mistralai"
            )

        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
        # Initialize client
        client = MistralClient(api_key=self.api_key)
        
        # Convert messages to Mistral format
        mistral_messages = []
        for msg in messages:
            mistral_messages.append(ChatMessage(
                role=msg["role"],
                content=msg["content"]
            ))
        
        try:
            # Make the API call
            response = client.chat(
                model=self.model,
                messages=mistral_messages,
                temperature=merged_chat_params.get("temperature", 0.7),
                max_tokens=merged_chat_params.get("max_tokens", 1024),
                top_p=merged_chat_params.get("top_p", 1.0),
                stream=False
            )
            
            # Extract the response content
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return ""
                
        except Exception as e:
            raise RuntimeError(f"Mistral API call failed: {e}")
