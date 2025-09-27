from __future__ import annotations

import json
import os
import requests
from typing import Any, Sequence, Dict, Optional
from urllib.parse import urljoin, urlparse
from pathlib import Path
import traceback

from .base import LLMProvider
from ..types import LLMMessage


class OllamaLLM(LLMProvider):
    """
    Ollama LLM provider for local Ollama server instances.
    Supports both local and remote Ollama servers.
    This class uses a singleton pattern to prevent re-initialization.
    """
    _instances: Dict[str, "OllamaLLM"] = {}

    def __new__(cls, *args, **kwargs):
        """Create a new instance or return the existing one."""
        model = kwargs.get('model', 'default_model')
        base_url = kwargs.get('base_url', 'default_url')
        instance_key = f"{model}-{base_url}"

        if instance_key not in cls._instances:
            instance = super(OllamaLLM, cls).__new__(cls)
            cls._instances[instance_key] = instance
        return cls._instances[instance_key]
    
    def __init__(self, model: str = None, base_url: str = None, **kwargs: Any) -> None:
        """
        Initialize Ollama LLM provider. This will only run once per unique instance.
        
        Args:
            model: Model name (e.g., "llama2", "codellama", "mistral")
            base_url: Base URL of the Ollama server (e.g., "http://localhost:11434")
            **kwargs: Additional parameters
        """
        if hasattr(self, '_initialized') and self._initialized:
            return

        from ..logging_utils import get_logger
        self.logger = get_logger()

        self.logger.info("🚀 INITIALIZING OLLAMA LLM PROVIDER (first time only)...")
        self.model = model
        self.base_url = self._initialize_base_url(base_url)
        
        # Ensure base_url doesn't end with /
        if self.base_url.endswith('/'):
            self.base_url = self.base_url[:-1]
        
        # Default parameters
        self.default_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 512),
            'top_p': kwargs.get('top_p', 0.9),
        }
        
        # Test connection
        self._initialize_client()
        self._initialized = True

    def _initialize_base_url(self, base_url: Optional[str] = None) -> str:
        """Initialize base URL with better configuration management."""
        if base_url:
            self.logger.info(f"💡 Using explicitly provided base URL: {base_url}")
            return base_url
        
        # Check environment variable
        env_url = os.getenv("OLLAMA_BASE_URL")
        if env_url:
            self.logger.info(f"💡 Using base URL from OLLAMA_BASE_URL environment variable: {env_url}")
            return env_url
        
        # Check for Ollama configuration file
        ollama_config_path = Path.home() / ".ollama" / "config.json"
        if ollama_config_path.exists():
            try:
                with open(ollama_config_path, 'r') as f:
                    config = json.load(f)
                    if 'base_url' in config:
                        self.logger.info(f"💡 Using base URL from Ollama configuration file: {config['base_url']}")
                        return config['base_url']
            except Exception as e:
                self.logger.warning(f"⚠️ Could not read Ollama config file: {e}")
        
        # Default to localhost
        default_url = "http://localhost:11434"
        self.logger.info(f"💡 Using default Ollama base URL: {default_url}")
        return default_url

    def _initialize_client(self) -> None:
        """Initialize the Ollama client with proper error handling."""
        self.logger.info("🔧 Initializing Ollama client...")
        try:
            self._test_connection()
            self.logger.info("✅ Ollama client initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ Error during Ollama client initialization: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize Ollama client: {e}")
    
    def _test_connection(self) -> None:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Ollama server at {self.base_url}: {e}")
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using the Ollama model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        if not hasattr(self, '_initialized') or not self._initialized:
            self.__init__(**kwargs)

        self.logger.info(f"💬 Ollama chat called with {len(messages)} messages")

        # Convert messages to Ollama format
        ollama_messages = []
        for message in messages:
            ollama_messages.append({
                "role": message["role"],
                "content": message["content"]
            })
        
        # Merge generation parameters
        gen_params = {**self.default_params, **kwargs}
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": gen_params.get('temperature', 0.7),
                "num_predict": gen_params.get('max_tokens', 512),
                "top_p": gen_params.get('top_p', 0.9),
            }
        }
        
        try:
            self.logger.info(f"🚀 Invoking Ollama model: {self.model}")
            
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            self.logger.info(f"✅ Successfully extracted response from model.")
            return result["message"]["content"].strip()
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ Ollama API request failed: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Ollama API request failed: {e}")
        except (KeyError, json.JSONDecodeError) as e:
            self.logger.error(f"❌ Invalid response from Ollama server: {e}")
            self.logger.debug(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Invalid response from Ollama server: {e}")
    
    def _format_messages(self, messages: Sequence[LLMMessage]) -> str:
        """Convert message sequence to prompt format (not used in Ollama)."""
        # Ollama handles message formatting internally
        return ""
