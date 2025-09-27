from __future__ import annotations

import os
from typing import Any, Sequence, Dict, Optional
from pathlib import Path
import traceback

from .base import LLMProvider
from ..types import LLMMessage


class CohereChat(LLMProvider):
    """
    Cohere LLM provider using the Cohere API.
    Supports Command, Command Light, Command Nightly, and other Cohere models.
    This class uses a singleton pattern to prevent re-initialization.
    """
    _instances: Dict[str, "CohereChat"] = {}

    def __new__(cls, *args, **kwargs):
        """Create a new instance or return the existing one."""
        model = kwargs.get('model', 'default_model')
        instance_key = f"{model}"

        if instance_key not in cls._instances:
            instance = super(CohereChat, cls).__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]
    
    def __init__(self, model: str = "command", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Cohere provider. This will only run once per unique instance.
        
        Args:
            model: Cohere model name (e.g., 'command', 'command-light', 'command-nightly')
            api_key: Cohere API key (defaults to COHERE_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        from ..logging_utils import get_logger
        self.logger = get_logger()

        self.logger.info("🚀 INITIALIZING COHERE LLM PROVIDER (first time only)...")
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
        env_key = os.getenv("COHERE_API_KEY")
        if env_key:
            self.logger.info("💡 Using API key from COHERE_API_KEY environment variable")
            return env_key
        
        # Check for Cohere configuration file
        cohere_config_path = Path.home() / ".cohere" / "config.json"
        if cohere_config_path.exists():
            try:
                import json
                with open(cohere_config_path, 'r') as f:
                    config = json.load(f)
                    if 'api_key' in config:
                        self.logger.info("💡 Using API key from Cohere configuration file")
                        return config['api_key']
            except Exception as e:
                self.logger.warning(f"⚠️ Could not read Cohere config file: {e}")
        
        # If no API key found, raise error with helpful message
        self.logger.error("❌ Cohere API key not found!")
        self.logger.error("   Please set COHERE_API_KEY environment variable or pass api_key parameter.")
        self.logger.error("   You can also create a config file at ~/.cohere/config.json")
        raise ValueError("Cohere API key is required. Set COHERE_API_KEY environment variable or pass api_key parameter.")

    def _initialize_client(self) -> None:
        """Initialize the Cohere client with proper error handling."""
        self.logger.info("🔧 Initializing Cohere client...")
        try:
            import cohere
        except ImportError:
            self.logger.error("❌ cohere package is required for Cohere but is not installed.")
            raise ImportError("cohere package is required. Install with: pip install cohere")

        try:
            # Initialize client
            self._client = cohere.Client(self.api_key, **self.client_kwargs)
            self.logger.info("✅ Cohere client created successfully")
            
            # Test the connection by making a simple API call
            self.logger.info("🔍 Testing Cohere connection...")
            test_response = self._client.chat(
                model=self.model,
                message="Hi",
                max_tokens=1
            )
            self.logger.info("✅ Successfully tested Cohere connection")
            
        except Exception as e:
            self.logger.error(f"❌ Error during Cohere client initialization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize Cohere client: {e}")
    
    def _validate_model(self) -> None:
        """Validate the model name format."""
        valid_models = [
            # Command R Series (Latest)
            "command-r-plus",
            "command-r",
            "command-r-16k",
            "command-r-32k",
            
            # Command Series
            "command",
            "command-light",
            "command-nightly",
            "command-light-16k",
            "command-nightly-16k",
            
            # Command R+ Series
            "command-r-plus-16k",
            "command-r-plus-32k",
            
            # Legacy Models
            "command-v1",
            "command-light-v1",
            "command-nightly-v1",
            
            # Future Models (when available)
            "command-r-2",
            "command-r-2-plus",
        ]
        
        if self.model not in valid_models:
            self.logger.warning(f"Warning: Model '{self.model}' may not be available. Valid models: {valid_models[:10]}...")

    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Cohere model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not hasattr(self, '_initialized') or not self._initialized:
            self.__init__(**kwargs)

        self.logger.info(f"💬 Cohere chat called with {len(messages)} messages")

        if self._client is None:
            raise RuntimeError("Cohere client not initialized")

        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
        # Convert messages to Cohere format
        # Cohere expects a single prompt string with conversation history
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
                conversation_parts.append(f"Human: {content}")
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")
        
        # Combine system message with conversation
        if system_message:
            prompt = f"System: {system_message}\n\n" + "\n".join(conversation_parts) + "\n\nAssistant:"
        else:
            prompt = "\n".join(conversation_parts) + "\n\nAssistant:"
        
        try:
            self.logger.info(f"🚀 Invoking Cohere model: {self.model}")
            
            # Make the API call
            response = self._client.chat(
                model=self.model,
                message=prompt,
                temperature=merged_chat_params.get("temperature", 0.7),
                max_tokens=merged_chat_params.get("max_tokens", 1024),
                p=merged_chat_params.get("top_p", 0.9),
                k=merged_chat_params.get("top_k", 0),
                stream=False
            )
            
            self.logger.info(f"✅ Successfully extracted response from model.")
            
            # Extract the response content
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                return response.message.content
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"❌ Cohere API call failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Cohere API call failed: {e}")
