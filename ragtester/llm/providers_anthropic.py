from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage
from .api_utils import retry_with_backoff, validate_api_key, validate_model_name, handle_api_response, validate_messages_format


class AnthropicChat(LLMProvider):
    """
    Anthropic Claude LLM provider using the official Anthropic API.
    Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, and other Claude models.
    """
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Anthropic Claude provider.
        
        Args:
            model: Claude model name (e.g., 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022')
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = validate_api_key(api_key or os.getenv("ANTHROPIC_API_KEY"), "Anthropic")
        
        # Store chat parameters separately from client parameters
        self.chat_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1024),
            'top_p': kwargs.get('top_p', 1.0)
        }
        
        # Store other parameters that might be used by the client
        self.client_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['temperature', 'max_tokens', 'top_p']}
        
        # API key validation is handled by validate_api_key
        
        # Validate model name
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        valid_models = [
            # Claude 3.5 Series (Latest)
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-5-opus-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            
            # Claude 3 Series
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            
            # Claude 2 Series
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
            "claude-instant-1.1",
            "claude-instant-1.0",
            
            # Claude 1 Series (Legacy)
            "claude-1.3",
            "claude-1.2",
            "claude-1.1",
            "claude-1.0",
            
            # Claude Instant Series
            "claude-instant-1.2",
            "claude-instant-1.1",
            "claude-instant-1.0",
            
            # Claude for AWS Bedrock
            "anthropic.claude-3-5-sonnet-20241022-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-2.1-v1:0",
            "anthropic.claude-2.0-v1:0",
            "anthropic.claude-instant-1.2-v1:0",
        ]
        
        validate_model_name(self.model, valid_models, "Anthropic")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Claude model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package is required. Install with: pip install anthropic"
            )

        # Validate messages format
        validate_messages_format(list(messages), "Anthropic")

        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
        # Initialize client with only client-specific parameters
        client = anthropic.Anthropic(api_key=self.api_key, **self.client_kwargs)
        
        # Separate system message and conversation messages
        system_message = ""
        conversation_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                # Anthropic expects assistant/user roles
                # Handle both text and multimodal content
                if isinstance(msg["content"], list):
                    # Multimodal content (for vision models)
                    conversation_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    # Text content
                    conversation_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Prepare parameters
        params = {
            "model": self.model,
            "max_tokens": merged_chat_params.get("max_tokens", 1024),
            "temperature": merged_chat_params.get("temperature", 0.7),
            "messages": conversation_messages
        }
        
        # Add system message if present
        if system_message:
            params["system"] = system_message
        
        # Add other parameters if specified
        if "top_p" in merged_chat_params:
            params["top_p"] = merged_chat_params["top_p"]
        if "top_k" in merged_chat_params:
            params["top_k"] = merged_chat_params["top_k"]
        if "stop_sequences" in merged_chat_params:
            params["stop_sequences"] = merged_chat_params["stop_sequences"]
        
        try:
            # Make the API call
            response = client.messages.create(**params)
            
            return handle_api_response(response, "Anthropic")
                
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")
