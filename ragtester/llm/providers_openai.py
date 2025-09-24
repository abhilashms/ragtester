from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage
from .api_utils import retry_with_backoff, validate_api_key, validate_model_name, handle_api_response, validate_messages_format


class OpenAIChat(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None, base_url: str = None, **kwargs: Any) -> None:
        self.model = model
        self.api_key = validate_api_key(api_key or os.getenv("OPENAI_API_KEY"), "OpenAI")
        self.base_url = base_url
        
        # Validate model name
        self._validate_model()
        
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
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        valid_models = [
            # GPT-4o Series (Latest)
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-2024-07-18",
            
            # GPT-4 Series
            "gpt-4",
            "gpt-4-0613",
            "gpt-4-0314",
            "gpt-4-32k",
            "gpt-4-32k-0613",
            "gpt-4-32k-0314",
            "gpt-4-turbo",
            "gpt-4-turbo-2024-04-09",
            "gpt-4-turbo-preview",
            
            # GPT-3.5 Series
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-0301",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-16k-0613",
            "gpt-3.5-turbo-1106",
            "gpt-3.5-turbo-0125",
            
            # GPT-3 Series (Legacy)
            "text-davinci-003",
            "text-davinci-002",
            "text-davinci-001",
            "text-curie-001",
            "text-babbage-001",
            "text-ada-001",
            
            # Embedding Models
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
            
            # Moderation Models
            "text-moderation-latest",
            "text-moderation-stable",
            
            # DALL-E Models
            "dall-e-3",
            "dall-e-2",
            
            # Whisper Models
            "whisper-1",
            
            # TTS Models
            "tts-1",
            "tts-1-hd",
            
            # Custom Models (if using OpenAI Custom)
            "gpt-4-custom",
            "gpt-3.5-turbo-custom",
        ]
        
        validate_model_name(self.model, valid_models, "OpenAI")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        try:
            import openai
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")

        # Validate messages format
        validate_messages_format(list(messages), "OpenAI")

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
        
        return handle_api_response(response, "OpenAI")
