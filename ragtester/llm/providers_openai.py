from __future__ import annotations

import os
from typing import Any, Sequence, Dict, Optional
from pathlib import Path
import traceback

from .base import LLMProvider
from ..types import LLMMessage
from .api_utils import retry_with_backoff, validate_api_key, validate_model_name, handle_api_response, validate_messages_format


class OpenAIChat(LLMProvider):
    """
    OpenAI LLM provider that supports various OpenAI models.
    This class uses a singleton pattern to prevent re-initialization.
    """
    _instances: Dict[str, "OpenAIChat"] = {}

    def __new__(cls, *args, **kwargs):
        """Create a new instance or return the existing one."""
        model = kwargs.get('model', 'default_model')
        base_url = kwargs.get('base_url', 'default_url')
        instance_key = f"{model}-{base_url}"

        if instance_key not in cls._instances:
            instance = super(OpenAIChat, cls).__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None, base_url: str = None, **kwargs: Any) -> None:
        """
        Initialize OpenAI LLM provider. This will only run once per unique instance.
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        from ..logging_utils import get_logger
        self.logger = get_logger()

        self.logger.info("🚀 INITIALIZING OPENAI LLM PROVIDER (first time only)...")
        self.model = model
        self.base_url = base_url
        
        # Initialize API key with better error handling
        self.api_key = self._initialize_api_key(api_key)
        
        # Validate model name
        self.model = self._validate_model()
        
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
                             if k not in ['temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'model', 'api_key', 'base_url']}
        
        self._client = None
        self._initialize_client()
        self._initialized = True

    def _initialize_api_key(self, api_key: Optional[str] = None) -> str:
        """Initialize API key with better error handling and credential management."""
        # Try to get API key from various sources
        if api_key:
            self.logger.info("💡 Using explicitly provided API key")
            return api_key
        
        # Check environment variable
        env_key = os.getenv("OPENAI_API_KEY")
        if env_key:
            self.logger.info("💡 Using API key from OPENAI_API_KEY environment variable")
            return env_key
        
        # Check for OpenAI configuration file
        openai_config_path = Path.home() / ".openai" / "config.json"
        if openai_config_path.exists():
            try:
                import json
                with open(openai_config_path, 'r') as f:
                    config = json.load(f)
                    if 'api_key' in config:
                        self.logger.info("💡 Using API key from OpenAI configuration file")
                        return config['api_key']
            except Exception as e:
                self.logger.warning(f"⚠️ Could not read OpenAI config file: {e}")
        
        # If no API key found, raise error with helpful message
        self.logger.error("❌ OpenAI API key not found!")
        self.logger.error("   Please set OPENAI_API_KEY environment variable or pass api_key parameter.")
        self.logger.error("   You can also create a config file at ~/.openai/config.json")
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client with proper error handling."""
        self.logger.info("🔧 Initializing OpenAI client...")
        try:
            import openai
        except ImportError:
            self.logger.error("❌ openai package is required for OpenAI but is not installed.")
            raise ImportError("openai package is required for OpenAI. Install with: pip install openai")

        try:
            # Prepare client arguments
            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url
            client_kwargs.update(self.client_kwargs)
            
            self._client = openai.OpenAI(**client_kwargs)
            self.logger.info("✅ OpenAI client created successfully")
            
            # Test the connection by listing models
            self.logger.info("🔍 Testing OpenAI connection by listing available models...")
            models = self._client.models.list()
            available_models = [model.id for model in models.data]
            self.logger.info(f"✅ Successfully listed {len(available_models)} models")
            
            if self.model not in available_models:
                self.logger.warning(f"⚠️ Model {self.model} not found in available models list. Proceeding with invocation attempt.")
                
        except Exception as e:
            self.logger.error(f"❌ Error during OpenAI client initialization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize OpenAI client: {e}")
    
    def _validate_model(self) -> str:
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
        
        return validate_model_name(self.model, valid_models, "OpenAI")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        if not hasattr(self, '_initialized') or not self._initialized:
            self.__init__(**kwargs)

        self.logger.info(f"💬 OpenAI chat called with {len(messages)} messages")
        validate_messages_format(list(messages), "OpenAI")

        if self._client is None:
            raise RuntimeError("OpenAI client not initialized")

        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
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
        
        try:
            self.logger.info(f"🚀 Invoking OpenAI model: {self.model}")
            
            # Make the API call
            response = self._client.chat.completions.create(
                model=self.model,
                messages=openai_messages,
                **merged_chat_params
            )
            
            self.logger.info(f"✅ Successfully extracted response from model.")
            return handle_api_response(response, "OpenAI")
            
        except Exception as e:
            self.logger.error(f"❌ OpenAI API call failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"OpenAI API call failed: {e}")
