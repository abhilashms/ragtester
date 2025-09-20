from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class CohereChat(LLMProvider):
    """
    Cohere LLM provider using the Cohere API.
    Supports Command, Command Light, Command Nightly, and other Cohere models.
    """
    
    def __init__(self, model: str = "command", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Cohere provider.
        
        Args:
            model: Cohere model name (e.g., 'command', 'command-light', 'command-nightly')
            api_key: Cohere API key (defaults to COHERE_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        
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
                "Cohere API key is required. Set COHERE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Validate model name
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        valid_models = [
            "command",
            "command-light",
            "command-nightly",
            "command-r",
            "command-r-plus",
            "command-r-16k",
            "command-light-16k",
            "command-nightly-16k",
            "command-r-32k"
        ]
        
        if self.model not in valid_models:
            print(f"Warning: Model '{self.model}' may not be available. Valid models: {valid_models}")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Cohere model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "cohere package is required. Install with: pip install cohere"
            )

        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
        # Initialize client
        client = cohere.Client(self.api_key)
        
        # Convert messages to Cohere format
        # Cohere expects a single prompt string with conversation history
        conversation_parts = []
        system_message = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_message = content
            elif role == "user":
                conversation_parts.append(f"Human: {content}")
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")
        
        # Combine system message with conversation
        if system_message:
            prompt = f"System: {system_message}\n\n" + "\n".join(conversation_parts) + "\n\nAssistant:"
        else:
            prompt = "\n".join(conversation_parts) + "\n\nAssistant:"
        
        try:
            # Make the API call
            response = client.chat(
                model=self.model,
                message=prompt,
                temperature=merged_kwargs.get("temperature", 0.7),
                max_tokens=merged_kwargs.get("max_tokens", 1024),
                p=merged_kwargs.get("top_p", 0.9),
                k=merged_kwargs.get("top_k", 0),
                stream=False
            )
            
            # Extract the response content
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                return response.message.content
            else:
                return ""
                
        except Exception as e:
            raise RuntimeError(f"Cohere API call failed: {e}")
