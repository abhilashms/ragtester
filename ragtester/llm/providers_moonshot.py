from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class MoonshotLLM(LLMProvider):
    """
    Moonshot AI LLM provider that supports Moonshot models via their API.
    """
    
    def __init__(self, model: str = "moonshot-v1-8k", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Moonshot LLM provider.
        
        Args:
            model: Moonshot model name (e.g., 'moonshot-v1-8k', 'moonshot-v1-32k', 'moonshot-v1-128k')
            api_key: Moonshot API key
            **kwargs: Additional parameters
        """
        # Import logging utilities
        from ..logging_utils import get_logger
        self.logger = get_logger()
        
        self.model = model
        self.api_key = api_key or os.getenv('MOONSHOT_API_KEY')
        
        if not self.api_key:
            raise ValueError("Moonshot API key is required. Set MOONSHOT_API_KEY environment variable or pass api_key parameter.")
        
        # Store generation parameters
        self.generation_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1024),
            'top_p': kwargs.get('top_p', 1.0),
            'stream': kwargs.get('stream', False)
        }
        
        self._client = None
        self._initialize_client()
        
        self.logger.debug(f"🚀 Initialized Moonshot LLM Provider")
        self.logger.debug(f"  - Model: {self.model}")
        self.logger.debug(f"  - Generation params: {self.generation_params}")
    
    def _initialize_client(self) -> None:
        """Initialize the Moonshot client."""
        self.logger.debug("🔧 Initializing Moonshot client...")
        
        try:
            import requests
            self.logger.debug("✅ requests imported successfully")
            
            # Moonshot uses REST API
            self.base_url = "https://api.moonshot.cn/v1"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            self.logger.debug("✅ Moonshot client initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"❌ Import error: {e}")
            self.logger.error("requests is required for Moonshot API.")
            self.logger.error("Install with: pip install requests")
            raise ImportError(
                "requests is required for Moonshot API. "
                "Install with: pip install requests"
            )
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during Moonshot client initialization: {e}")
            raise
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Moonshot model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        self.logger.debug(f"💬 Moonshot chat called with {len(messages)} messages")
        
        # Merge generation parameters with any additional kwargs
        merged_params = {**self.generation_params, **kwargs}
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": merged_params.get('temperature', 0.7),
            "max_tokens": merged_params.get('max_tokens', 1024),
            "top_p": merged_params.get('top_p', 1.0),
            "stream": merged_params.get('stream', False)
        }
        
        self.logger.debug(f"Request payload: {payload}")
        
        try:
            import requests
            
            self.logger.debug(f"🚀 Calling Moonshot API...")
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            self.logger.debug("✅ Moonshot API call successful")
            
            # Extract response text
            if 'choices' in response_data and len(response_data['choices']) > 0:
                response_text = response_data['choices'][0]['message']['content']
                self.logger.debug(f"✅ Extracted response: {response_text[:100]}...")
                return response_text
            else:
                self.logger.error("❌ No response content in API response")
                raise RuntimeError("No response content in Moonshot API response")
                
        except Exception as e:
            self.logger.error(f"❌ Moonshot API call failed: {e}")
            raise RuntimeError(f"Moonshot API call failed: {e}")
