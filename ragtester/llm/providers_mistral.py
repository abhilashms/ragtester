from __future__ import annotations

import os
from typing import Any, Sequence, Dict, Optional
from pathlib import Path
import traceback

from .base import LLMProvider
from ..types import LLMMessage


class MistralChat(LLMProvider):
    """
    Mistral AI LLM provider using the Mistral API.
    Supports Mistral Large, Mistral Medium, Mistral Small, and other Mistral models.
    This class uses a singleton pattern to prevent re-initialization.
    """
    _instances: Dict[str, "MistralChat"] = {}

    def __new__(cls, *args, **kwargs):
        """Create a new instance or return the existing one."""
        model = kwargs.get('model', 'default_model')
        instance_key = f"{model}"

        if instance_key not in cls._instances:
            instance = super(MistralChat, cls).__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]
    
    def __init__(self, model: str = "mistral-large-latest", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Mistral provider. This will only run once per unique instance.
        
        Args:
            model: Mistral model name (e.g., 'mistral-large-latest', 'mistral-medium-latest', 'mistral-small-latest')
            api_key: Mistral API key (defaults to MISTRAL_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        from ..logging_utils import get_logger
        self.logger = get_logger()

        self.logger.info("🚀 INITIALIZING MISTRAL LLM PROVIDER (first time only)...")
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
        self._validate_model()
        
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
        env_key = os.getenv("MISTRAL_API_KEY")
        if env_key:
            self.logger.info("💡 Using API key from MISTRAL_API_KEY environment variable")
            return env_key
        
        # Check for Mistral configuration file
        mistral_config_path = Path.home() / ".mistral" / "config.json"
        if mistral_config_path.exists():
            try:
                import json
                with open(mistral_config_path, 'r') as f:
                    config = json.load(f)
                    if 'api_key' in config:
                        self.logger.info("💡 Using API key from Mistral configuration file")
                        return config['api_key']
            except Exception as e:
                self.logger.warning(f"⚠️ Could not read Mistral config file: {e}")
        
        # If no API key found, raise error with helpful message
        self.logger.error("❌ Mistral API key not found!")
        self.logger.error("   Please set MISTRAL_API_KEY environment variable or pass api_key parameter.")
        self.logger.error("   You can also create a config file at ~/.mistral/config.json")
        raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass api_key parameter.")

    def _initialize_client(self) -> None:
        """Initialize the Mistral client with proper error handling."""
        self.logger.info("🔧 Initializing Mistral client...")
        try:
            from mistralai.client import MistralClient
        except ImportError:
            self.logger.error("❌ mistralai package is required for Mistral but is not installed.")
            raise ImportError("mistralai package is required. Install with: pip install mistralai")

        try:
            # Initialize client
            self._client = MistralClient(api_key=self.api_key, **self.client_kwargs)
            self.logger.info("✅ Mistral client created successfully")
            
            # Test the connection by making a simple API call
            self.logger.info("🔍 Testing Mistral connection...")
            from mistralai.models.chat_completion import ChatMessage
            test_response = self._client.chat(
                model=self.model,
                messages=[ChatMessage(role="user", content="Hi")],
                max_tokens=1
            )
            self.logger.info("✅ Successfully tested Mistral connection")
            
        except Exception as e:
            self.logger.error(f"❌ Error during Mistral client initialization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize Mistral client: {e}")
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        valid_models = [
            # Mistral Large Series (Latest)
            "mistral-large-latest",
            "mistral-large-2402",
            
            # Mistral Medium Series
            "mistral-medium-latest",
            "mistral-medium-2312",
            
            # Mistral Small Series
            "mistral-small-latest",
            "mistral-small-2402",
            
            # Mistral 7B Series
            "mistral-7b-instruct",
            "mistral-7b-instruct-v0.1",
            "mistral-7b-instruct-v0.2",
            "mistral-7b-instruct-v0.3",
            
            # Mixtral Series (Mixture of Experts)
            "mistral-8x7b-instruct",
            "mistral-8x7b-instruct-v0.1",
            "mistral-8x22b-instruct",
            "mistral-8x22b-instruct-v0.1",
            
            # Codestral Series (Code Generation)
            "codestral-latest",
            "codestral-22b-2405",
            "codestral-7b-2405",
            
            # Pixtral Series (Multimodal)
            "pixtral-latest",
            "pixtral-12b-2409",
            
            # Future Models (when available)
            "mistral-nemo-latest",
            "mistral-nemo-12b-2409",
        ]
        
        if self.model not in valid_models:
            self.logger.warning(f"Warning: Model '{self.model}' may not be available. Valid models: {valid_models[:10]}...")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Mistral model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not hasattr(self, '_initialized') or not self._initialized:
            self.__init__(**kwargs)

        self.logger.info(f"💬 Mistral chat called with {len(messages)} messages")

        if self._client is None:
            raise RuntimeError("Mistral client not initialized")

        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
        # Convert messages to Mistral format
        from mistralai.models.chat_completion import ChatMessage
        mistral_messages = []
        for msg in messages:
            mistral_messages.append(ChatMessage(
                role=msg["role"],
                content=msg["content"]
            ))
        
        try:
            self.logger.info(f"🚀 Invoking Mistral model: {self.model}")
            
            # Make the API call
            response = self._client.chat(
                model=self.model,
                messages=mistral_messages,
                temperature=merged_chat_params.get("temperature", 0.7),
                max_tokens=merged_chat_params.get("max_tokens", 1024),
                top_p=merged_chat_params.get("top_p", 1.0),
                stream=False
            )
            
            self.logger.info(f"✅ Successfully extracted response from model.")
            
            # Extract the response content
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"❌ Mistral API call failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Mistral API call failed: {e}")
