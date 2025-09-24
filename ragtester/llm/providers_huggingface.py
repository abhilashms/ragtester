from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage
from .api_utils import (
    retry_with_backoff, validate_messages_format,
    handle_api_error, handle_rate_limit_response, test_api_connection,
    AuthenticationError, RateLimitError, PermissionError, ServerError,
    ConnectionError, TimeoutError
)


class HuggingFaceChat(LLMProvider):
    """
    Hugging Face Inference API provider.
    Supports thousands of open-source models including Llama, Mistral, CodeLlama, and more.
    """
    
    def __init__(self, model: str = "microsoft/DialoGPT-medium", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Hugging Face provider.
        
        Args:
            model: Hugging Face model name (e.g., 'microsoft/DialoGPT-medium', 'meta-llama/Llama-2-7b-chat-hf')
            api_key: Hugging Face API key (defaults to HF_TOKEN environment variable)
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_API_KEY")
        self.kwargs = kwargs
        
        if not self.api_key:
            raise ValueError(
                "Hugging Face API key is required. Set HF_TOKEN environment variable "
                "or pass api_key parameter."
            )
        
        # Validate model name
        self._validate_model()
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        # Popular chat models on Hugging Face
        popular_models = [
            "microsoft/DialoGPT-medium",
            "microsoft/DialoGPT-large", 
            "facebook/blenderbot-400M-distill",
            "facebook/blenderbot-1B-distill",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Llama-2-13b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "microsoft/Orca-2-7b",
            "microsoft/Orca-2-13b",
            "HuggingFaceH4/zephyr-7b-beta",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO"
        ]
        
        if self.model not in popular_models:
            print(f"Info: Using custom model '{self.model}'. Popular models: {popular_models[:5]}...")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Hugging Face Inference API.
        
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

        # Merge kwargs
        merged_kwargs = {**self.kwargs, **kwargs}
        
        # Convert messages to prompt format
        prompt = self._format_messages(messages)
        
        # Prepare request payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": merged_kwargs.get("max_tokens", 512),
                "temperature": merged_kwargs.get("temperature", 0.7),
                "top_p": merged_kwargs.get("top_p", 0.9),
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        # Add optional parameters
        if "top_k" in merged_kwargs:
            payload["parameters"]["top_k"] = merged_kwargs["top_k"]
        if "repetition_penalty" in merged_kwargs:
            payload["parameters"]["repetition_penalty"] = merged_kwargs["repetition_penalty"]
        
        # Prepare headers
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Make the API call to Hugging Face Inference API
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{self.model}",
                json=payload,
                headers=headers,
                timeout=60
            )
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse response
            response_data = response.json()
            
            # Extract the response content
            if isinstance(response_data, list) and len(response_data) > 0:
                # Most HF models return a list with generated text
                if "generated_text" in response_data[0]:
                    return response_data[0]["generated_text"].strip()
                elif "text" in response_data[0]:
                    return response_data[0]["text"].strip()
            elif isinstance(response_data, dict):
                # Some models return a dict
                if "generated_text" in response_data:
                    return response_data["generated_text"].strip()
                elif "text" in response_data:
                    return response_data["text"].strip()
            
            return ""
                
        except requests.exceptions.RequestException as e:
            # Use enhanced error handling
            handle_api_error(e, "HuggingFace", f"Model: {self.model}")
        except Exception as e:
            # Use enhanced error handling
            handle_api_error(e, "HuggingFace", f"Model: {self.model}")
    
    def _format_messages(self, messages: Sequence[LLMMessage]) -> str:
        """Convert message sequence to prompt format for Hugging Face models."""
        # Format conversation for most HF chat models
        conversation_parts = []
        system_message = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # Handle both text and multimodal content for system messages
                if isinstance(content, str):
                    system_message = content
                else:
                    # For multimodal content, convert to string representation
                    system_message = str(content)
            elif role == "user":
                conversation_parts.append(f"User: {content}")
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")
        
        # Combine system message with conversation
        if system_message:
            prompt = f"System: {system_message}\n\n" + "\n".join(conversation_parts) + "\nAssistant:"
        else:
            prompt = "\n".join(conversation_parts) + "\nAssistant:"
        
        return prompt
