from __future__ import annotations

import os
from typing import Any, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class GeminiChat(LLMProvider):
    """
    Google Gemini LLM provider using the Google AI API.
    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash, and other Gemini models.
    """
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Gemini provider.
        
        Args:
            model: Gemini model name (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp')
            api_key: Google AI API key (defaults to GOOGLE_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        self.model = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
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
                "Google AI API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Validate model name
        self._validate_model()
        
        # Initialize the client
        self._initialize_client()
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        valid_models = [
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-2.0-flash-exp",
            "gemini-pro",
            "gemini-pro-vision"
        ]
        
        if self.model not in valid_models:
            print(f"Warning: Model '{self.model}' may not be available. Valid models: {valid_models}")
    
    def _initialize_client(self) -> None:
        """Initialize the Google AI client."""
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required. Install with: pip install google-generativeai"
            )
        
        # Configure the API key
        genai.configure(api_key=self.api_key)
        
        # Initialize the model
        self._genai = genai
        self._client = genai.GenerativeModel(self.model, **self.client_kwargs)

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Gemini model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not hasattr(self, '_client'):
            self._initialize_client()
        
        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
        # Convert messages to Gemini format
        # Gemini uses a different message format than OpenAI
        conversation_history = []
        system_instruction = ""
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                conversation_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                conversation_history.append({"role": "model", "parts": [content]})
        
        try:
            # Create generation config
            generation_config = self._genai.types.GenerationConfig(
                max_output_tokens=merged_chat_params.get("max_tokens", 1024),
                temperature=merged_chat_params.get("temperature", 0.7),
                top_p=merged_chat_params.get("top_p", 0.95),
                top_k=merged_chat_params.get("top_k", 40),
            )
            
            # If we have conversation history, use chat mode
            if conversation_history:
                # Start a new chat session
                chat = self._client.start_chat(history=conversation_history[:-1])
                
                # Get the last user message
                last_message = conversation_history[-1]["parts"][0]
                
                # Generate response
                response = chat.send_message(
                    last_message,
                    generation_config=generation_config
                )
            else:
                # Single message generation
                # Use the system instruction if available, otherwise use the first message content
                if system_instruction:
                    prompt = f"{system_instruction}\n\n{messages[-1]['content'] if messages else ''}"
                else:
                    prompt = messages[-1]["content"] if messages else ""
                
                response = self._client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            
            # Extract the response text
            if response and response.text:
                return response.text
            else:
                return ""
                
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")
