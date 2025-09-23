"""
Custom exceptions for RAG testing framework.
Provides structured error handling with proper categorization.
"""

from typing import Any, Dict, Optional


class RAGTesterError(Exception):
    """Base exception for all RAG testing framework errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.original_error = original_error
    
    def __str__(self) -> str:
        base_msg = self.message
        if self.error_code:
            base_msg = f"[{self.error_code}] {base_msg}"
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            base_msg = f"{base_msg} (Context: {context_str})"
        return base_msg


class ConfigurationError(RAGTesterError):
    """Raised when there are configuration-related issues."""
    pass


class LLMError(RAGTesterError):
    """Base exception for LLM-related errors."""
    pass


class LLMProviderError(LLMError):
    """Raised when LLM provider initialization or operation fails."""
    pass


class LLMRequestError(LLMError):
    """Raised when LLM request fails (network, API, etc.)."""
    pass


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid or cannot be parsed."""
    pass


class RAGClientError(RAGTesterError):
    """Base exception for RAG client-related errors."""
    pass


class RAGClientConnectionError(RAGClientError):
    """Raised when RAG client cannot connect to the service."""
    pass


class RAGClientTimeoutError(RAGClientError):
    """Raised when RAG client request times out."""
    pass


class RAGClientResponseError(RAGClientError):
    """Raised when RAG client returns an invalid response."""
    pass


class DocumentProcessingError(RAGTesterError):
    """Base exception for document processing errors."""
    pass


class DocumentLoadError(DocumentProcessingError):
    """Raised when document loading fails."""
    pass


class DocumentParseError(DocumentProcessingError):
    """Raised when document parsing fails."""
    pass


class DocumentValidationError(DocumentProcessingError):
    """Raised when document validation fails."""
    pass


class QuestionGenerationError(RAGTesterError):
    """Base exception for question generation errors."""
    pass


class QuestionGenerationTimeoutError(QuestionGenerationError):
    """Raised when question generation times out."""
    pass


class QuestionValidationError(QuestionGenerationError):
    """Raised when generated question validation fails."""
    pass


class EvaluationError(RAGTesterError):
    """Base exception for evaluation errors."""
    pass


class EvaluationTimeoutError(EvaluationError):
    """Raised when evaluation times out."""
    pass


class EvaluationResponseError(EvaluationError):
    """Raised when evaluation response is invalid."""
    pass


class ValidationError(RAGTesterError):
    """Raised when input validation fails."""
    pass


class NetworkError(RAGTesterError):
    """Raised when network-related operations fail."""
    pass


class TimeoutError(RAGTesterError):
    """Raised when operations exceed their timeout limits."""
    pass


class ResourceError(RAGTesterError):
    """Raised when resource-related operations fail (memory, disk, etc.)."""
    pass


class RetryableError(RAGTesterError):
    """Base exception for errors that can be retried."""
    pass


class NonRetryableError(RAGTesterError):
    """Base exception for errors that should not be retried."""
    pass


def wrap_exception(original_error: Exception, message: str, 
                  error_code: Optional[str] = None, 
                  context: Optional[Dict[str, Any]] = None) -> RAGTesterError:
    """
    Wrap a generic exception in a RAG testing framework exception.
    
    Args:
        original_error: The original exception to wrap
        message: Human-readable error message
        error_code: Optional error code for categorization
        context: Optional context information
        
    Returns:
        Appropriate RAGTesterError subclass based on the original error type
    """
    error_type = type(original_error).__name__
    
    # Map common exception types to our custom exceptions
    if isinstance(original_error, (ConnectionError, OSError)):
        return NetworkError(message, error_code, context, original_error)
    elif isinstance(original_error, TimeoutError):
        return TimeoutError(message, error_code, context, original_error)
    elif isinstance(original_error, (ValueError, TypeError)):
        return ValidationError(message, error_code, context, original_error)
    elif isinstance(original_error, (FileNotFoundError, PermissionError)):
        return ResourceError(message, error_code, context, original_error)
    elif isinstance(original_error, (KeyError, AttributeError)):
        return ConfigurationError(message, error_code, context, original_error)
    else:
        return RAGTesterError(message, error_code, context, original_error)


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error is retryable based on its type and context.
    
    Args:
        error: The exception to check
        
    Returns:
        True if the error is retryable, False otherwise
    """
    if isinstance(error, RetryableError):
        return True
    elif isinstance(error, NonRetryableError):
        return False
    elif isinstance(error, (NetworkError, TimeoutError, RAGClientTimeoutError, 
                           LLMRequestError, QuestionGenerationTimeoutError, 
                           EvaluationTimeoutError)):
        return True
    elif isinstance(error, (ValidationError, ConfigurationError, 
                           DocumentValidationError, QuestionValidationError)):
        return False
    else:
        # Default to retryable for unknown errors
        return True
