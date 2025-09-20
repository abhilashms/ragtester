from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from .base import LLMProvider
from ..types import LLMMessage
from .logging_wrapper import LoggingLLMWrapper


class DummyLLM(LLMProvider):
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        # Very naive echo-style model for offline testing
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[DUMMY ANSWER] {last_user[:400]}"


def build_llm(provider_name: str, **kwargs: Any) -> LLMProvider:
    name = (provider_name or "").lower()
    model_name = kwargs.get('model', 'unknown')
    
    if name in ("", "dummy"):
        provider = DummyLLM()
        return LoggingLLMWrapper(provider, "dummy", model_name)
    
    try:
        if name == "openai":
            from .providers_openai import OpenAIChat
            provider = OpenAIChat(**kwargs)
        elif name == "anthropic":
            from .providers_anthropic import AnthropicChat
            provider = AnthropicChat(**kwargs)
        elif name == "grok":
            from .providers_grok import GrokChat
            provider = GrokChat(**kwargs)
        elif name == "gemini":
            from .providers_gemini import GeminiChat
            provider = GeminiChat(**kwargs)
        elif name == "mistral":
            from .providers_mistral import MistralChat
            provider = MistralChat(**kwargs)
        elif name == "cohere":
            from .providers_cohere import CohereChat
            provider = CohereChat(**kwargs)
        elif name == "huggingface":
            from .providers_huggingface import HuggingFaceChat
            provider = HuggingFaceChat(**kwargs)
        elif name == "fireworks":
            from .providers_fireworks import FireworksChat
            provider = FireworksChat(**kwargs)
        elif name == "together":
            from .providers_together import TogetherChat
            provider = TogetherChat(**kwargs)
        elif name == "perplexity":
            from .providers_perplexity import PerplexityChat
            provider = PerplexityChat(**kwargs)
        elif name == "local":
            from .providers_local import LocalLLM
            provider = LocalLLM(**kwargs)
        elif name == "bedrock":
            from .providers_bedrock import BedrockLLM
            provider = BedrockLLM(**kwargs)
        else:
            raise ValueError(f"Unknown provider: {name}")
        
        # Wrap with logging
        return LoggingLLMWrapper(provider, name, model_name)
        
    except Exception as e:
        # Log the error and fallback to dummy if provider not installed or fails to init
        import logging
        logger = logging.getLogger("ragtester")
        logger.error(f"Failed to initialize {name} provider: {e}")
        logger.warning(f"Falling back to dummy provider. Check your configuration and dependencies.")
        
        provider = DummyLLM()
        return LoggingLLMWrapper(provider, "dummy", model_name)


