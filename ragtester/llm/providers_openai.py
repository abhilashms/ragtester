from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class OpenAIChat(LLMProvider):
    def __init__(self, model: str = "gpt-3.5-turbo", api_key: str = None, base_url: str = None, **kwargs: Any) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url
        
        # Store chat parameters separately from client parameters
        self.chat_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1024),
            'top_p': kwargs.get('top_p', 1.0),
            'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
            'presence_penalty': kwargs.get('presence_penalty', 0.0)
        }
        
        # Store other parameters that might be used by the client
        self.client_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty']}
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
        # Prepare the client
        client_kwargs = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        # Add any additional client parameters
        client_kwargs.update(self.client_kwargs)
        
        client = openai.OpenAI(api_key=self.api_key, **client_kwargs)
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            # Handle both text and multimodal content
            if isinstance(msg["content"], list):
                # Multimodal content (for vision models)
                openai_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            else:
                # Text content
                openai_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Make the API call
        response = client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            **merged_chat_params
        )
        
        return response.choices[0].message.content
