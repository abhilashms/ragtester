from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence

from .base import LLMProvider
from ..types import LLMMessage


class BaiduLLM(LLMProvider):
    """
    Baidu ERNIE LLM provider that supports Baidu ERNIE models via their API.
    """
    
    def __init__(self, model: str = "ernie-bot", api_key: str = None, **kwargs: Any) -> None:
        """
        Initialize Baidu ERNIE LLM provider.
        
        Args:
            model: Baidu model name (e.g., 'ernie-bot', 'ernie-bot-turbo', 'ernie-bot-4')
            api_key: Baidu API key
            **kwargs: Additional parameters
        """
        # Import logging utilities
        from ..logging_utils import get_logger
        self.logger = get_logger()
        
        self.model = model
        self.api_key = api_key or os.getenv('BAIDU_API_KEY')
        
        if not self.api_key:
            raise ValueError("Baidu API key is required. Set BAIDU_API_KEY environment variable or pass api_key parameter.")
        
        # Store generation parameters
        self.generation_params = {
            'temperature': kwargs.get('temperature', 0.7),
            'max_tokens': kwargs.get('max_tokens', 1024),
            'top_p': kwargs.get('top_p', 1.0),
            'stream': kwargs.get('stream', False)
        }
        
        self._client = None
        self._initialize_client()
        
        self.logger.debug(f"🚀 Initialized Baidu ERNIE LLM Provider")
        self.logger.debug(f"  - Model: {self.model}")
        self.logger.debug(f"  - Generation params: {self.generation_params}")
    
    def _initialize_client(self) -> None:
        """Initialize the Baidu client."""
        self.logger.debug("🔧 Initializing Baidu client...")
        
        try:
            import requests
            self.logger.debug("✅ requests imported successfully")
            
            # Baidu uses REST API
            self.base_url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
            self.headers = {
                "Content-Type": "application/json"
            }
            
            self.logger.debug("✅ Baidu client initialized successfully")
            
        except ImportError as e:
            self.logger.error(f"❌ Import error: {e}")
            self.logger.error("requests is required for Baidu API.")
            self.logger.error("Install with: pip install requests")
            raise ImportError(
                "requests is required for Baidu API. "
                "Install with: pip install requests"
            )
        except Exception as e:
            self.logger.error(f"❌ Unexpected error during Baidu client initialization: {e}")
            raise
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """
        Generate a response using Baidu ERNIE model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response text
        """
        self.logger.debug(f"💬 Baidu ERNIE chat called with {len(messages)} messages")
        
        # Merge generation parameters with any additional kwargs
        merged_params = {**self.generation_params, **kwargs}
        
        # Convert messages to Baidu format
        baidu_messages = []
        for msg in messages:
            if msg["role"] == "user":
                baidu_messages.append({"role": "user", "content": msg["content"]})
            elif msg["role"] == "assistant":
                baidu_messages.append({"role": "assistant", "content": msg["content"]})
            elif msg["role"] == "system":
                # Baidu doesn't have system messages, prepend to first user message
                if baidu_messages and baidu_messages[0]["role"] == "user":
                    baidu_messages[0]["content"] = f"System: {msg['content']}\n\n{baidu_messages[0]['content']}"
                else:
                    baidu_messages.append({"role": "user", "content": f"System: {msg['content']}"})
        
        # Prepare request payload
        payload = {
            "messages": baidu_messages,
            "temperature": merged_params.get('temperature', 0.7),
            "max_output_tokens": merged_params.get('max_tokens', 1024),
            "top_p": merged_params.get('top_p', 1.0),
            "stream": merged_params.get('stream', False)
        }
        
        self.logger.debug(f"Request payload: {payload}")
        
        try:
            import requests
            
            # Baidu uses access_token in URL
            url = f"{self.base_url}/{self.model}?access_token={self.api_key}"
            
            self.logger.debug(f"🚀 Calling Baidu ERNIE API...")
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            self.logger.debug("✅ Baidu ERNIE API call successful")
            
            # Extract response text
            if 'result' in response_data:
                response_text = response_data['result']
                self.logger.debug(f"✅ Extracted response: {response_text[:100]}...")
                return response_text
            else:
                self.logger.error("❌ No response content in API response")
                raise RuntimeError("No response content in Baidu ERNIE API response")
                
        except Exception as e:
            self.logger.error(f"❌ Baidu ERNIE API call failed: {e}")
            raise RuntimeError(f"Baidu ERNIE API call failed: {e}")
