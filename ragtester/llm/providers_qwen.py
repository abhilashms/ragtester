from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class QwenLLM(LLMProvider):
    """
    Qwen (Alibaba) LLM provider that supports Qwen models via their API.
    """
    
    def __init__(self, model: str = "qwen-plus", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Qwen LLM provider.
        
        Args:
            model: Qwen model name (e.g., 'qwen-plus', 'qwen-max', 'qwen-turbo')
            api_key: Qwen API key
            **kwargs: Additional parameters
        """
        # Import logging utilities
        from ..logging_utils import get_logger
        self.logger = get_logger()
        
        self.model = model
        self.api_key = api_key or os.getenv('QWEN_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
        
        if not self.api_key:
            raise ValueError("Qwen API key is required. Set QWEN_API_KEY or DASHSCOPE_API_KEY environment variable or pass api_key parameter.")
        
        # Store generation parameters
        self.generation_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1024),
            'top_p': kwargs.get('top_p', 1.0),
            'stream': kwargs.get('stream', False)
        }
        
        self._client = None
        self._initialize_client()
        
        self.logger.debug(f"🚀 Initialized Qwen LLM Provider")
        self.logger.debug(f"  - Model: {self.model}")
        self.logger.debug(f"  - Generation params: {self.generation_params}")
    
    def _initialize_client(self) -> None:
        """Initialize the Qwen client."""
        self.logger.debug("🔧 Initializing Qwen client...")
        
        try:
            import requests
            self.logger.debug("✅ requests imported successfully")
            
            # Qwen uses DashScope API
            self.base_url = "https://dashscope.aliyuncs.com/api/v1"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            self.logger.debug("✅ Qwen client initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"❌ Import error: {e}")
            self.logger.error("requests is required for Qwen API.")
            self.logger.error("Install with: pip install requests")
            raise ImportError(
                "requests is required for Qwen API. "
                "Install with: pip install requests"
            )
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during Qwen client initialization: {e}")
            raise
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Qwen model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        self.logger.debug(f"💬 Qwen chat called with {len(messages)} messages")
        
        # Merge generation parameters with any additional kwargs
        merged_params = {**self.generation_params, **kwargs}
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "input": {
                "messages": list(messages)
            },
            "parameters": {
                "temperature": merged_params.get('temperature', 0.7),
                "max_tokens": merged_params.get('max_tokens', 1024),
                "top_p": merged_params.get('top_p', 1.0)
            }
        }
        
        self.logger.debug(f"Request payload: {payload}")
        
        try:
            import requests
            
            self.logger.debug(f"🚀 Calling Qwen API...")
            response = requests.post(
                f"{self.base_url}/services/aigc/text-generation/generation",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            self.logger.debug("✅ Qwen API call successful")
            
            # Extract response text
            if 'output' in response_data and 'text' in response_data['output']:
                response_text = response_data['output']['text']
                self.logger.debug(f"✅ Extracted response: {response_text[:100]}...")
                return response_text
            else:
                self.logger.error("❌ No response content in API response")
                raise RuntimeError("No response content in Qwen API response")
                
        except Exception as e:
            self.logger.error(f"❌ Qwen API call failed: {e}")
            raise RuntimeError(f"Qwen API call failed: {e}")
