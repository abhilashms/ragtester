from __future__ import annotations

import os
from typing import Any, Sequence, Dict, Optional
from pathlib import Path
import traceback

from .base import LLMProvider
from ..types import LLMMessage
from .api_utils import retry_with_backoff, validate_api_key, validate_model_name, handle_api_response, validate_messages_format


class GeminiChat(LLMProvider):
    """
    Google Gemini LLM provider using the Google AI API.
    Supports Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 2.0 Flash, and other Gemini models.
    This class uses a singleton pattern to prevent re-initialization.
    """
    _instances: Dict[str, "GeminiChat"] = {}

    def __new__(cls, *args, **kwargs):
        """Create a new instance or return the existing one."""
        model = kwargs.get('model', 'default_model')
        instance_key = f"{model}"

        if instance_key not in cls._instances:
            instance = super(GeminiChat, cls).__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]
    
    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Gemini provider. This will only run once per unique instance.
        
        Args:
            model: Gemini model name (e.g., 'gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-2.0-flash-exp')
            api_key: Google AI API key (defaults to GOOGLE_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        from ..logging_utils import get_logger
        self.logger = get_logger()

        self.logger.info("🚀 INITIALIZING GEMINI LLM PROVIDER (first time only)...")
        self.model = model
        
        # Initialize API key with better error handling
        self.api_key = self._initialize_api_key(api_key)
        
        # Store chat parameters separately from client parameters
        self.chat_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1024),
            'top_p': kwargs.get('top_p', 1.0)
        }
        
        # Store other parameters that might be used by the client
        self.client_kwargs = {k: v for k, v in kwargs.items() 
                             if k not in ['temperature', 'max_tokens', 'top_p', 'model', 'api_key']}
        
        # Validate model name
        self.model = self._validate_model()
        
        # Initialize the client
        self._initialize_client()
        self._initialized = True

    def _initialize_api_key(self, api_key: Optional[str] = None) -> str:
        """Initialize API key with better error handling and credential management."""
        # Try to get API key from various sources
        if api_key:
            self.logger.info("💡 Using explicitly provided API key")
            return api_key
        
        # Check environment variable
        env_key = os.getenv("GOOGLE_API_KEY")
        if env_key:
            self.logger.info("💡 Using API key from GOOGLE_API_KEY environment variable")
            return env_key
        
        # Check for Google configuration file
        google_config_path = Path.home() / ".google" / "config.json"
        if google_config_path.exists():
            try:
                import json
                with open(google_config_path, 'r') as f:
                    config = json.load(f)
                    if 'api_key' in config:
                        self.logger.info("💡 Using API key from Google configuration file")
                        return config['api_key']
            except Exception as e:
                self.logger.warning(f"⚠️ Could not read Google config file: {e}")
        
        # If no API key found, raise error with helpful message
        self.logger.error("❌ Google API key not found!")
        self.logger.error("   Please set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        self.logger.error("   You can also create a config file at ~/.google/config.json")
        raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
    
    def _validate_model(self) -> str:
        """Validate the model name format."""
        valid_models = [
            # Gemini 2.0 Series (Latest)
            "gemini-2.0-flash-exp",
            "gemini-2.0-flash-thinking-exp",
            
            # Gemini 1.5 Series
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-pro-001",
            "gemini-1.5-flash-001",
            
            # Gemini 1.0 Series
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-pro-001",
            "gemini-pro-vision-001",
            
            # Gemini Ultra Series
            "gemini-ultra",
            "gemini-ultra-vision",
            
            # Experimental Models
            "gemini-experimental",
            "gemini-experimental-vision",
            
            # Future Models (when available)
            "gemini-3.0-flash",
            "gemini-3.0-pro",
        ]
        
        return validate_model_name(self.model, valid_models, "Google Gemini")
    
    def _initialize_client(self) -> None:
        """Initialize the Google AI client with proper error handling."""
        self.logger.info("🔧 Initializing Gemini client...")
        try:
            import google.generativeai as genai
        except ImportError:
            self.logger.error("❌ google-generativeai package is required for Gemini but is not installed.")
            raise ImportError(
                "google-generativeai package is required. Install with: pip install google-generativeai"
            )
        
        try:
            # Configure the API key
            genai.configure(api_key=self.api_key)
            
            # Initialize the model
            self._genai = genai
            self._client = genai.GenerativeModel(self.model, **self.client_kwargs)
            self.logger.info("✅ Gemini client created successfully")
            
            # Test the connection by making a simple API call
            self.logger.info("🔍 Testing Gemini connection...")
            test_response = self._client.generate_content("Hi")
            self.logger.info("✅ Successfully tested Gemini connection")
            
        except Exception as e:
            self.logger.error(f"❌ Error during Gemini client initialization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize Gemini client: {e}")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Gemini model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not hasattr(self, '_initialized') or not self._initialized:
            self.__init__(**kwargs)

        self.logger.info(f"💬 Gemini chat called with {len(messages)} messages")
        validate_messages_format(list(messages), "Google Gemini")

        if self._client is None:
            raise RuntimeError("Gemini client not initialized")
        
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
                # Handle both text and multimodal content for system messages
                if isinstance(content, str):
                    system_instruction = content
                else:
                    # For multimodal content, convert to string representation
                    system_instruction = str(content)
            elif role == "user":
                conversation_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                conversation_history.append({"role": "model", "parts": [content]})
        
        try:
            self.logger.info(f"🚀 Invoking Gemini model: {self.model}")
            
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
            
            self.logger.info(f"✅ Successfully extracted response from model.")
            return handle_api_response(response, "Google Gemini")
                
        except Exception as e:
            self.logger.error(f"❌ Gemini API call failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Gemini API call failed: {e}")
