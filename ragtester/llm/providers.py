from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class DummyLLM(LLMProvider):
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        # Very naive echo-style model for offline testing
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"[DUMMY ANSWER] {last_user[:400]}"


def build_llm(provider_name: str, **kwargs: Any) -> LLMProvider:
    name = (provider_name or "").lower()
    if name in ("", "dummy"):
        return DummyLLM()
    try:
        if name == "openai":
            from .providers_openai import OpenAIChat
            return OpenAIChat(**kwargs)
        if name == "anthropic":
            from .providers_anthropic import AnthropicChat
            return AnthropicChat(**kwargs)
        if name == "grok":
            from .providers_grok import GrokChat
            return GrokChat(**kwargs)
        if name == "gemini":
            from .providers_gemini import GeminiChat
            return GeminiChat(**kwargs)
        if name == "mistral":
            from .providers_mistral import MistralChat
            return MistralChat(**kwargs)
        if name == "cohere":
            from .providers_cohere import CohereChat
            return CohereChat(**kwargs)
        if name == "huggingface":
            from .providers_huggingface import HuggingFaceChat
            return HuggingFaceChat(**kwargs)
        if name == "fireworks":
            from .providers_fireworks import FireworksChat
            return FireworksChat(**kwargs)
        if name == "together":
            from .providers_together import TogetherChat
            return TogetherChat(**kwargs)
        if name == "perplexity":
            from .providers_perplexity import PerplexityChat
            return PerplexityChat(**kwargs)
        if name == "local":
            from .providers_local import LocalLLM
            return LocalLLM(**kwargs)
        if name == "bedrock":
            from .providers_bedrock import BedrockLLM
            return BedrockLLM(**kwargs)
    except Exception:
        # Fallback to dummy if provider not installed or fails to init
        return DummyLLM()
    return DummyLLM()


