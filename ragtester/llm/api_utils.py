"""
Utility functions for API providers to ensure robust functionality.
"""

import time
import random
from typing import Any, Dict, Callable, Optional
from functools import wraps


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,)
):
    """
    Decorator to retry API calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        # Last attempt failed, raise the exception
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)
                    
                    time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def validate_api_key(api_key: Optional[str], provider_name: str) -> str:
    """
    Validate that an API key is provided and not empty.
    
    Args:
        api_key: The API key to validate
        provider_name: Name of the provider for error messages
        
    Returns:
        The validated API key
        
    Raises:
        ValueError: If API key is missing or empty
    """
    if not api_key:
        raise ValueError(
            f"{provider_name} API key is required. "
            f"Set {provider_name.upper()}_API_KEY environment variable or pass api_key parameter."
        )
    
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError(f"{provider_name} API key must be a non-empty string.")
    
    return api_key.strip()


def validate_model_name(model: str, valid_models: list, provider_name: str) -> str:
    """
    Validate and normalize model name for the provider.
    
    Args:
        model: The model name to validate
        valid_models: List of valid model names
        provider_name: Name of the provider for error messages
        
    Returns:
        The normalized model name
    """
    # Import logging utilities
    from ..logging_utils import get_logger
    logger = get_logger()
    
    # Normalize model name for AWS Bedrock
    if provider_name == "AWS Bedrock":
        normalized_model = _normalize_bedrock_model_name(model)
        if normalized_model != model:
            logger.info(f"🔧 Normalized Bedrock model name: '{model}' → '{normalized_model}'")
            model = normalized_model
    
    if model not in valid_models:
        logger.warning(
            f"⚠️ Model '{model}' may not be available for {provider_name}. "
            f"Valid models: {valid_models[:5]}..."
        )
    
    return model


def _normalize_bedrock_model_name(model: str) -> str:
    """
    Normalize AWS Bedrock model names by removing region prefixes.
    
    Args:
        model: The model name to normalize
        
    Returns:
        The normalized model name
    """
    # Remove common AWS region prefixes
    region_prefixes = ['us.', 'eu.', 'ap.', 'ca.', 'sa.', 'af.', 'me.']
    
    for prefix in region_prefixes:
        if model.startswith(prefix):
            normalized = model[len(prefix):]
            # Import logging utilities
            from ..logging_utils import get_logger
            logger = get_logger()
            logger.debug(f"Removed region prefix '{prefix}' from model name: '{model}' → '{normalized}'")
            return normalized
    
    return model


def handle_api_response(response: Any, provider_name: str) -> str:
    """
    Handle API response and extract text content.
    
    Args:
        response: The API response object
        provider_name: Name of the provider for error messages
        
    Returns:
        Extracted text content
        
    Raises:
        RuntimeError: If response cannot be processed
    """
    # Import logging utilities
    from ..logging_utils import get_logger
    logger = get_logger()
    
    try:
        # Handle different response formats
        if hasattr(response, 'choices') and response.choices:
            # OpenAI-style response
            return response.choices[0].message.content
        elif hasattr(response, 'content') and response.content:
            # Anthropic-style response
            if isinstance(response.content, list) and len(response.content) > 0:
                return response.content[0].text
            elif isinstance(response.content, str):
                return response.content
        elif hasattr(response, 'text'):
            # Direct text response
            return response.text
        elif hasattr(response, 'message'):
            # Message-based response
            if hasattr(response.message, 'content'):
                return response.message.content
            elif hasattr(response.message, 'text'):
                return response.message.text
        elif isinstance(response, str):
            # String response
            return response
        elif isinstance(response, dict):
            # Dictionary response - try common keys
            for key in ['text', 'content', 'message', 'response', 'output']:
                if key in response:
                    content = response[key]
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, dict) and 'text' in content:
                        return content['text']
        
        logger.error(f"❌ Unable to extract text from {provider_name} response: {type(response)}")
        raise RuntimeError(f"Unable to extract text from {provider_name} API response")
        
    except Exception as e:
        logger.error(f"❌ Error processing {provider_name} response: {e}")
        raise RuntimeError(f"Error processing {provider_name} API response: {e}")


def create_http_headers(api_key: str, provider_name: str, custom_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Create standard HTTP headers for API requests.
    
    Args:
        api_key: The API key
        provider_name: Name of the provider
        custom_headers: Additional custom headers
        
    Returns:
        Dictionary of HTTP headers
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": f"RAGtester/1.0 ({provider_name})"
    }
    
    if custom_headers:
        headers.update(custom_headers)
    
    return headers


def validate_messages_format(messages: list, provider_name: str) -> None:
    """
    Validate that messages are in the correct format.
    
    Args:
        messages: List of message dictionaries
        provider_name: Name of the provider for error messages
        
    Raises:
        ValueError: If messages format is invalid
    """
    if not isinstance(messages, list):
        raise ValueError(f"{provider_name}: messages must be a list")
    
    if not messages:
        raise ValueError(f"{provider_name}: messages list cannot be empty")
    
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise ValueError(f"{provider_name}: message {i} must be a dictionary")
        
        if "role" not in msg:
            raise ValueError(f"{provider_name}: message {i} must have 'role' field")
        
        if "content" not in msg:
            raise ValueError(f"{provider_name}: message {i} must have 'content' field")
        
        if msg["role"] not in ["system", "user", "assistant"]:
            raise ValueError(f"{provider_name}: message {i} role must be 'system', 'user', or 'assistant'")
        
        if not isinstance(msg["content"], (str, list)):
            raise ValueError(f"{provider_name}: message {i} content must be string or list")
