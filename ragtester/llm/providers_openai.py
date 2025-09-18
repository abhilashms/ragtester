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
        self.kwargs = kwargs
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        # Merge kwargs
        merged_kwargs = {**self.kwargs, **kwargs}
        
        # Prepare the client
        client_kwargs = {}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        
        client = openai.OpenAI(api_key=self.api_key, **client_kwargs)
        
        # Convert messages to OpenAI format
        openai_messages = []
        for msg in messages:
            openai_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Make the API call
        response = client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            **merged_kwargs
        )
        
        return response.choices[0].message.content
