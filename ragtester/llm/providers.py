from __future__ import annotations

import os
from typing import Any, Dict, Sequence

from .base import LLMProvider
from ..types import LLMMessage
from .logging_wrapper import LoggingLLMWrapper


class DummyLLM(LLMProvider):
    def __init__(self):
        # Import logging utilities
        from ..logging_utils import get_logger
        self.logger = get_logger()
        self.logger.debug("[WARN] DummyLLM initialized - this is a fallback provider for testing")
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        # Very naive echo-style model for offline testing
        self.logger.debug("[WARN] DummyLLM.chat() called - returning dummy response")
        self.logger.debug(f"DummyLLM received {len(messages)} messages")
        
        last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        dummy_response = f"[DUMMY ANSWER] {last_user[:400]}"
        
        self.logger.debug(f"[WARN] DummyLLM returning: {dummy_response[:100]}...")
        return dummy_response


def build_llm(provider_name: str, **kwargs: Any) -> LLMProvider:
    name = (provider_name or "").lower()
    model_name = kwargs.get('model', 'unknown')
    
    # Import logging utilities
    from ..logging_utils import get_logger
    logger = get_logger()
    
    logger.debug(f"[BUILD] Building LLM - Provider: {name}, Model: {model_name}")
    logger.debug(f"LLM kwargs: {kwargs}")
    
    if name in ("", "dummy"):
        logger.debug("Using dummy provider (explicitly requested)")
        provider = DummyLLM()
        return LoggingLLMWrapper(provider, "dummy", model_name)
    
    logger.debug(f"Attempting to initialize {name} provider...")
    
    try:
        if name == "openai":
            logger.debug("Importing OpenAI provider...")
            from .providers_openai import OpenAIChat
            # OpenAI expects: model, api_key, base_url, and other kwargs
            provider = OpenAIChat(**kwargs)
            logger.debug("[OK] OpenAI provider initialized successfully")
        elif name == "anthropic":
            logger.debug("Importing Anthropic provider...")
            from .providers_anthropic import AnthropicChat
            # Anthropic expects: model, api_key, and other kwargs (no base_url)
            anthropic_kwargs = {k: v for k, v in kwargs.items() if k != 'base_url'}
            logger.debug(f"Filtered Anthropic kwargs: {anthropic_kwargs}")
            provider = AnthropicChat(**anthropic_kwargs)
            logger.debug("[OK] Anthropic provider initialized successfully")
        elif name == "grok":
            logger.debug("Importing Grok provider...")
            from .providers_grok import GrokChat
            # Grok expects: model, api_key, and other kwargs (no base_url)
            grok_kwargs = {k: v for k, v in kwargs.items() if k != 'base_url'}
            logger.debug(f"Filtered Grok kwargs: {grok_kwargs}")
            provider = GrokChat(**grok_kwargs)
            logger.debug("[OK] Grok provider initialized successfully")
        elif name == "gemini":
            logger.debug("Importing Gemini provider...")
            from .providers_gemini import GeminiChat
            # Gemini expects: model, api_key, and other kwargs (no base_url)
            gemini_kwargs = {k: v for k, v in kwargs.items() if k != 'base_url'}
            logger.debug(f"Filtered Gemini kwargs: {gemini_kwargs}")
            provider = GeminiChat(**gemini_kwargs)
            logger.debug("[OK] Gemini provider initialized successfully")
        elif name == "mistral":
            logger.debug("Importing Mistral provider...")
            from .providers_mistral import MistralChat
            # Mistral expects: model, api_key, and other kwargs (no base_url)
            mistral_kwargs = {k: v for k, v in kwargs.items() if k != 'base_url'}
            logger.debug(f"Filtered Mistral kwargs: {mistral_kwargs}")
            provider = MistralChat(**mistral_kwargs)
            logger.debug("[OK] Mistral provider initialized successfully")
        elif name == "cohere":
            logger.debug("Importing Cohere provider...")
            from .providers_cohere import CohereChat
            # Cohere expects: model, api_key, and other kwargs (no base_url)
            cohere_kwargs = {k: v for k, v in kwargs.items() if k != 'base_url'}
            logger.debug(f"Filtered Cohere kwargs: {cohere_kwargs}")
            provider = CohereChat(**cohere_kwargs)
            logger.debug("[OK] Cohere provider initialized successfully")
        elif name == "huggingface":
            logger.debug("Importing HuggingFace provider...")
            from .providers_huggingface import HuggingFaceChat
            # HuggingFace expects: model, api_key, and other kwargs (no base_url)
            hf_kwargs = {k: v for k, v in kwargs.items() if k != 'base_url'}
            logger.debug(f"Filtered HuggingFace kwargs: {hf_kwargs}")
            provider = HuggingFaceChat(**hf_kwargs)
            logger.debug("[OK] HuggingFace provider initialized successfully")
        elif name == "fireworks":
            logger.debug("Importing Fireworks provider...")
            from .providers_fireworks import FireworksChat
            # Fireworks expects: model, api_key, and other kwargs (no base_url)
            fireworks_kwargs = {k: v for k, v in kwargs.items() if k != 'base_url'}
            logger.debug(f"Filtered Fireworks kwargs: {fireworks_kwargs}")
            provider = FireworksChat(**fireworks_kwargs)
            logger.debug("[OK] Fireworks provider initialized successfully")
        elif name == "together":
            logger.debug("Importing Together provider...")
            from .providers_together import TogetherChat
            # Together expects: model, api_key, and other kwargs (no base_url)
            together_kwargs = {k: v for k, v in kwargs.items() if k != 'base_url'}
            logger.debug(f"Filtered Together kwargs: {together_kwargs}")
            provider = TogetherChat(**together_kwargs)
            logger.debug("[OK] Together provider initialized successfully")
        elif name == "perplexity":
            logger.debug("Importing Perplexity provider...")
            from .providers_perplexity import PerplexityChat
            # Perplexity expects: model, api_key, and other kwargs (no base_url)
            perplexity_kwargs = {k: v for k, v in kwargs.items() if k != 'base_url'}
            logger.debug(f"Filtered Perplexity kwargs: {perplexity_kwargs}")
            provider = PerplexityChat(**perplexity_kwargs)
            logger.debug("[OK] Perplexity provider initialized successfully")
        elif name == "local":
            logger.debug("Importing Local provider...")
            from .providers_local import LocalLLM
            # Local expects: model and other kwargs (no api_key, base_url)
            local_kwargs = {k: v for k, v in kwargs.items() if k not in ['api_key', 'base_url']}
            logger.debug(f"Filtered Local kwargs: {local_kwargs}")
            provider = LocalLLM(**local_kwargs)
            logger.debug("[OK] Local provider initialized successfully")
        elif name == "ollama":
            logger.debug("Importing Ollama provider...")
            from .providers_ollama import OllamaLLM
            # Ollama expects: model, base_url, and other kwargs (no api_key)
            ollama_kwargs = {k: v for k, v in kwargs.items() if k != 'api_key'}
            logger.debug(f"Filtered Ollama kwargs: {ollama_kwargs}")
            provider = OllamaLLM(**ollama_kwargs)
            logger.debug("[OK] Ollama provider initialized successfully")
        elif name == "bedrock":
            logger.debug("Importing Bedrock provider...")
            from .providers_bedrock import BedrockLLM
            logger.debug(f"Initializing Bedrock with model: {model_name}, region: {kwargs.get('region', 'us-east-1')}")
            
            # Bedrock expects: model, region, and other kwargs (no api_key, base_url)
            bedrock_kwargs = {k: v for k, v in kwargs.items() if k not in ['api_key', 'base_url']}
            logger.debug(f"Filtered Bedrock kwargs: {bedrock_kwargs}")
            
            provider = BedrockLLM(**bedrock_kwargs)
            logger.debug("[OK] Bedrock provider initialized successfully")
        else:
            logger.error(f"[ERROR] Unknown provider: {name}")
            raise ValueError(f"Unknown provider: {name}")
        
        logger.debug(f"[OK] {name} provider created successfully, wrapping with logging...")
        # Wrap with logging
        wrapped_provider = LoggingLLMWrapper(provider, name, model_name)
        logger.debug(f"[OK] LLM build completed successfully for {name}")
        return wrapped_provider
        
    except ImportError as e:
        logger.error(f"[ERROR] Import error for {name} provider: {e}")
        logger.error(f"Missing dependency for {name}. Please install required packages.")
        logger.warning(f"Falling back to dummy provider due to import error.")
        
        provider = DummyLLM()
        return LoggingLLMWrapper(provider, "dummy", model_name)
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize {name} provider: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error details: {str(e)}")
        logger.warning(f"Falling back to dummy provider. Check your configuration and dependencies.")
        
        # Log additional debugging info
        if name == "bedrock":
            logger.error("[DEBUG] Bedrock-specific debugging info:")
            logger.error(f"  - Model: {model_name}")
            logger.error(f"  - Region: {kwargs.get('region', 'us-east-1')}")
            logger.error(f"  - All kwargs: {kwargs}")
            logger.error("  - Check: AWS credentials, boto3 installation, IAM permissions")
        
        provider = DummyLLM()
        return LoggingLLMWrapper(provider, "dummy", model_name)


