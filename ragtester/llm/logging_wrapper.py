"""
Logging wrapper for LLM providers to add comprehensive logging.
"""

import time
from typing import Any, Sequence
from .base import LLMProvider
from ..types import LLMMessage
from ..logging_utils import get_logger


class LoggingLLMWrapper(LLMProvider):
    """Wrapper around LLM providers that adds comprehensive logging."""
    
    def __init__(self, provider: LLMProvider, provider_name: str, model_name: str):
        self.provider = provider
        self.provider_name = provider_name
        self.model_name = model_name
        self.logger = get_logger()
    
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        """Chat with the LLM and log all interactions."""
        start_time = time.time()
        
        # Log the request
        self.logger.log_llm_request(
            self.provider_name,
            self.model_name,
            list(messages),
            **kwargs
        )
        
        try:
            # Make the actual call
            response = self.provider.chat(messages, **kwargs)
            
            # Calculate response time
            response_time = time.time() - start_time
            
            # Log the response
            self.logger.log_llm_response(response, response_time)
            
            # Check for dummy responses
            if "[DUMMY" in str(response).upper() or "DUMMY" in str(response).upper():
                self.logger.warning(f"⚠️ LLM returned dummy response: '{response}'")
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            self.logger.error(f"❌ LLM call failed after {response_time:.2f}s: {e}")
            raise
