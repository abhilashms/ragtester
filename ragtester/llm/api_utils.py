"""
Utility functions for API providers to ensure robust functionality.
"""

import time
import random
from typing import Any, Dict, Callable, Optional
from functools import wraps


# Custom exception classes for better error handling
class AuthenticationError(Exception):
    """Raised when authentication fails."""
    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded."""
    pass


class PermissionError(Exception):
    """Raised when access is forbidden."""
    pass


class ServerError(Exception):
    """Raised when server returns 5xx error."""
    pass


class ConnectionError(Exception):
    """Raised when network connection fails."""
    pass


class TimeoutError(Exception):
    """Raised when request times out."""
    pass


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    exceptions: tuple = (Exception,),
    rate_limit_errors: tuple = (RateLimitError,),
    retryable_errors: tuple = (ConnectionError, TimeoutError, ServerError)
):
    """
    Enhanced decorator to retry API calls with intelligent error handling.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        exceptions: Tuple of general exceptions to catch and retry
        rate_limit_errors: Tuple of rate limit exceptions (longer delays)
        retryable_errors: Tuple of retryable network/server errors
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except rate_limit_errors as e:
                    # Handle rate limits with longer delays
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(max_delay, base_delay * (exponential_base ** attempt) * 2)  # Double delay for rate limits
                        if jitter:
                            delay *= (0.5 + random.random() * 0.5)
                        
                        # Import logging utilities
                        from ..logging_utils import get_logger
                        logger = get_logger()
                        logger.warning(f"⏳ Rate limited. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}")
                        time.sleep(delay)
                        continue
                except retryable_errors as e:
                    # Handle retryable errors
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(max_delay, base_delay * (exponential_base ** attempt))
                        if jitter:
                            delay *= (0.5 + random.random() * 0.5)
                        
                        # Import logging utilities
                        from ..logging_utils import get_logger
                        logger = get_logger()
                        logger.warning(f"🔄 Retryable error. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}: {e}")
                        time.sleep(delay)
                        continue
                except exceptions as e:
                    # Handle general exceptions
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(max_delay, base_delay * (exponential_base ** attempt))
                        if jitter:
                            delay *= (0.5 + random.random() * 0.5)
                        
                        # Import logging utilities
                        from ..logging_utils import get_logger
                        logger = get_logger()
                        logger.warning(f"🔄 Error occurred. Waiting {delay:.1f}s before retry {attempt + 1}/{max_retries}: {e}")
                        time.sleep(delay)
                        continue
                except Exception as e:
                    # Non-retryable errors - raise immediately
                    raise e
            
            # All retries exhausted
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


def handle_api_error(error: Exception, provider_name: str, context: str = "") -> None:
    """
    Enhanced error handling with specific error types.
    
    Args:
        error: The exception that occurred
        provider_name: Name of the provider
        context: Additional context about where the error occurred
    """
    # Import logging utilities
    from ..logging_utils import get_logger
    logger = get_logger()
    
    # Handle HTTP errors from requests
    if hasattr(error, 'response') and hasattr(error.response, 'status_code'):
        status_code = error.response.status_code
        if status_code == 401:
            logger.error(f"❌ {provider_name} authentication failed: Invalid API key")
            raise AuthenticationError(f"Invalid API key for {provider_name}: {context}")
        elif status_code == 429:
            logger.error(f"⏳ {provider_name} rate limit exceeded")
            raise RateLimitError(f"Rate limit exceeded for {provider_name}: {context}")
        elif status_code == 403:
            logger.error(f"🚫 {provider_name} access forbidden")
            raise PermissionError(f"Access forbidden for {provider_name}: {context}")
        elif status_code == 404:
            logger.error(f"🔍 {provider_name} resource not found")
            raise RuntimeError(f"Resource not found for {provider_name}: {context}")
        elif status_code >= 500:
            logger.error(f"🔥 {provider_name} server error ({status_code})")
            raise ServerError(f"Server error ({status_code}) for {provider_name}: {context}")
        else:
            logger.error(f"❌ {provider_name} HTTP error ({status_code})")
            raise RuntimeError(f"HTTP error ({status_code}) for {provider_name}: {context}")
    
    # Handle network errors
    import requests
    if isinstance(error, requests.exceptions.ConnectionError):
        logger.error(f"🌐 {provider_name} connection failed")
        raise ConnectionError(f"Network connection failed for {provider_name}: {context}")
    elif isinstance(error, requests.exceptions.Timeout):
        logger.error(f"⏱️ {provider_name} request timeout")
        raise TimeoutError(f"Request timeout for {provider_name}: {context}")
    elif isinstance(error, requests.exceptions.SSLError):
        logger.error(f"🔒 {provider_name} SSL error")
        raise ConnectionError(f"SSL error for {provider_name}: {context}")
    elif isinstance(error, requests.exceptions.RequestException):
        logger.error(f"📡 {provider_name} request error")
        raise ConnectionError(f"Request error for {provider_name}: {context}")
    
    # Handle specific provider errors
    if "boto3" in str(error).lower() or "bedrock" in str(error).lower():
        if "credentials" in str(error).lower():
            logger.error(f"🔑 {provider_name} AWS credentials error")
            raise AuthenticationError(f"AWS credentials error for {provider_name}: {context}")
        elif "permission" in str(error).lower() or "access denied" in str(error).lower():
            logger.error(f"🚫 {provider_name} AWS permission error")
            raise PermissionError(f"AWS permission error for {provider_name}: {context}")
    
    # Generic fallback
    logger.error(f"❌ {provider_name} unexpected error: {error}")
    raise RuntimeError(f"Unexpected error for {provider_name}: {error}")


def handle_rate_limit_response(response, provider_name: str) -> bool:
    """
    Handle rate limiting response with exponential backoff.
    
    Args:
        response: The HTTP response object
        provider_name: Name of the provider
        
    Returns:
        True if rate limited and handled, False otherwise
    """
    if hasattr(response, 'status_code') and response.status_code == 429:
        # Import logging utilities
        from ..logging_utils import get_logger
        logger = get_logger()
        
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            wait_time = int(retry_after)
        else:
            wait_time = 60  # Default wait time
        
        logger.warning(f"⏳ {provider_name} rate limited. Waiting {wait_time}s...")
        time.sleep(wait_time)
        return True
    return False


def test_api_connection(provider_name: str, test_func: Callable, *args, **kwargs) -> bool:
    """
    Test API connection during initialization.
    
    Args:
        provider_name: Name of the provider
        test_func: Function to test the connection
        *args: Arguments for the test function
        **kwargs: Keyword arguments for the test function
        
    Returns:
        True if connection successful, False otherwise
    """
    try:
        test_func(*args, **kwargs)
        # Import logging utilities
        from ..logging_utils import get_logger
        logger = get_logger()
        logger.info(f"✅ {provider_name} connection test successful")
        return True
    except Exception as e:
        # Import logging utilities
        from ..logging_utils import get_logger
        logger = get_logger()
        logger.warning(f"⚠️ {provider_name} connection test failed: {e}")
        return False
