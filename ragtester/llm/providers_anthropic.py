from __future__ import annotations

import os
from typing import Any, Sequence, Dict, Optional
from pathlib import Path
import traceback

from .base import LLMProvider
from ..types import LLMMessage
from .api_utils import retry_with_backoff, validate_api_key, validate_model_name, handle_api_response, validate_messages_format


class AnthropicChat(LLMProvider):
    """
    Anthropic Claude LLM provider using the official Anthropic API.
    Supports Claude 3.5 Sonnet, Claude 3.5 Haiku, and other Claude models.
    This class uses a singleton pattern to prevent re-initialization.
    """
    _instances: Dict[str, "AnthropicChat"] = {}

    def __new__(cls, *args, **kwargs):
        """Create a new instance or return the existing one."""
        model = kwargs.get('model', 'default_model')
        instance_key = f"{model}"

        if instance_key not in cls._instances:
            instance = super(AnthropicChat, cls).__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]
    
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Anthropic Claude provider. This will only run once per unique instance.
        
        Args:
            model: Claude model name (e.g., 'claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022')
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY environment variable)
            **kwargs: Additional parameters
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        from ..logging_utils import get_logger
        self.logger = get_logger()

        self.logger.info("🚀 INITIALIZING ANTHROPIC LLM PROVIDER (first time only)...")
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
        env_key = os.getenv("ANTHROPIC_API_KEY")
        if env_key:
            self.logger.info("💡 Using API key from ANTHROPIC_API_KEY environment variable")
            return env_key
        
        # Check for Anthropic configuration file
        anthropic_config_path = Path.home() / ".anthropic" / "config.json"
        if anthropic_config_path.exists():
            try:
                import json
                with open(anthropic_config_path, 'r') as f:
                    config = json.load(f)
                    if 'api_key' in config:
                        self.logger.info("💡 Using API key from Anthropic configuration file")
                        return config['api_key']
            except Exception as e:
                self.logger.warning(f"⚠️ Could not read Anthropic config file: {e}")
        
        # If no API key found, raise error with helpful message
        self.logger.error("❌ Anthropic API key not found!")
        self.logger.error("   Please set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
        self.logger.error("   You can also create a config file at ~/.anthropic/config.json")
        raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")

    def _initialize_client(self) -> None:
        """Initialize the Anthropic client with proper error handling."""
        self.logger.info("🔧 Initializing Anthropic client...")
        try:
            import anthropic
        except ImportError:
            self.logger.error("❌ anthropic package is required for Anthropic but is not installed.")
            raise ImportError("anthropic package is required for Anthropic. Install with: pip install anthropic")

        try:
            # Initialize client with only client-specific parameters
            self._client = anthropic.Anthropic(api_key=self.api_key, **self.client_kwargs)
            self.logger.info("✅ Anthropic client created successfully")
            
            # Test the connection by making a simple API call
            self.logger.info("🔍 Testing Anthropic connection...")
            # Note: Anthropic doesn't have a models.list() endpoint, so we'll test with a simple completion
            test_response = self._client.messages.create(
                model=self.model,
                max_tokens=1,
                messages=[{"role": "user", "content": "Hi"}]
            )
            self.logger.info("✅ Successfully tested Anthropic connection")
            
        except Exception as e:
            self.logger.error(f"❌ Error during Anthropic client initialization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize Anthropic client: {e}")
    
    def _validate_model(self) -> str:
        """Validate the model name format."""
        valid_models = [
            # Claude 3.5 Series (Latest)
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022", 
            "claude-3-5-opus-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-haiku-20241022",
            
            # Claude 3 Series
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            
            # Claude 2 Series
            "claude-2.1",
            "claude-2.0",
            "claude-instant-1.2",
            "claude-instant-1.1",
            "claude-instant-1.0",
            
            # Claude 1 Series (Legacy)
            "claude-1.3",
            "claude-1.2",
            "claude-1.1",
            "claude-1.0",
            
            # Claude Instant Series
            "claude-instant-1.2",
            "claude-instant-1.1",
            "claude-instant-1.0",
            
            # Claude for AWS Bedrock
            "anthropic.claude-3-5-sonnet-20241022-v1:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            "anthropic.claude-2.1-v1:0",
            "anthropic.claude-2.0-v1:0",
            "anthropic.claude-instant-1.2-v1:0",
        ]
        
        return validate_model_name(self.model, valid_models, "Anthropic")

    @retry_with_backoff(max_retries=3, exceptions=(Exception,))
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Claude model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not hasattr(self, '_initialized') or not self._initialized:
            self.__init__(**kwargs)

        self.logger.info(f"💬 Anthropic chat called with {len(messages)} messages")
        validate_messages_format(list(messages), "Anthropic")

        if self._client is None:
            raise RuntimeError("Anthropic client not initialized")

        # Merge chat parameters with any additional kwargs from the call
        merged_chat_params = {**self.chat_params, **kwargs}
        
        # Separate system message and conversation messages
        system_message = ""
        conversation_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                # Handle both text and multimodal content for system messages
                content = msg["content"]
                if isinstance(content, str):
                    system_message = content
                else:
                    # For multimodal content, convert to string representation
                    system_message = str(content)
            else:
                # Anthropic expects assistant/user roles
                # Handle both text and multimodal content
                if isinstance(msg["content"], list):
                    # Multimodal content (for vision models)
                    conversation_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                else:
                    # Text content
                    conversation_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
        
        # Prepare parameters
        params = {
            "model": self.model,
            "max_tokens": merged_chat_params.get("max_tokens", 1024),
            "temperature": merged_chat_params.get("temperature", 0.7),
            "messages": conversation_messages
        }
        
        # Add system message if present
        if system_message:
            params["system"] = system_message
        
        # Add other parameters if specified
        if "top_p" in merged_chat_params:
            params["top_p"] = merged_chat_params["top_p"]
        if "top_k" in merged_chat_params:
            params["top_k"] = merged_chat_params["top_k"]
        if "stop_sequences" in merged_chat_params:
            params["stop_sequences"] = merged_chat_params["stop_sequences"]
        
        try:
            self.logger.info(f"🚀 Invoking Anthropic model: {self.model}")
            
            # Make the API call
            response = self._client.messages.create(**params)
            
            self.logger.info(f"✅ Successfully extracted response from model.")
            return handle_api_response(response, "Anthropic")
                
        except Exception as e:
            self.logger.error(f"❌ Anthropic API call failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Anthropic API call failed: {e}")
