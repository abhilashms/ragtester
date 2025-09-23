"""
Error handling utilities for RAG testing framework.
Provides decorators, context managers, and utilities for robust error handling.
"""

import functools
import time
import traceback
from typing import Any, Callable, Dict, Optional, Type, Union, List
from contextlib import contextmanager

from .exceptions import (
    RAGTesterError, RetryableError, NonRetryableError, 
    wrap_exception, is_retryable_error
)
from .logging_utils import get_logger


class ErrorHandler:
    """Centralized error handling with logging and recovery strategies."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger()
    
    def handle_error(self, error: Exception, operation: str, 
                    context: Optional[Dict[str, Any]] = None,
                    reraise: bool = True) -> Optional[Any]:
        """
        Handle an error with logging and optional recovery.
        
        Args:
            error: The exception to handle
            operation: Name of the operation that failed
            context: Additional context information
            reraise: Whether to reraise the error after handling
            
        Returns:
            Recovery value if available, None otherwise
        """
        context = context or {}
        
        # Log the error with full context
        self.logger.log_error(operation, error, **context)
        
        # Log stack trace for debugging
        if self.logger.config and self.logger.config.level == "DEBUG":
            self.logger.debug(f"Stack trace for {operation}: {traceback.format_exc()}")
        
        # Determine if error is retryable
        retryable = is_retryable_error(error)
        self.logger.debug(f"Error in {operation} is {'retryable' if retryable else 'non-retryable'}")
        
        if reraise:
            raise error
        
        return None


def retry_on_error(
    max_retries: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    on_retry: Optional[Callable] = None
):
    """
    Decorator to retry a function on specific exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff_factor: Multiplier for delay after each retry
        exceptions: Tuple of exception types to retry on
        on_retry: Optional callback function called before each retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == max_retries:
                        logger.error(f"Function {func.__name__} failed after {max_retries} retries: {e}")
                        raise
                    
                    logger.warning(f"Function {func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                    
                    if on_retry:
                        try:
                            on_retry(attempt, e, current_delay)
                        except Exception as retry_error:
                            logger.warning(f"Retry callback failed: {retry_error}")
                    
                    if attempt < max_retries:
                        logger.debug(f"Retrying {func.__name__} in {current_delay:.2f} seconds...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator


def handle_errors(
    operation: str,
    default_return: Any = None,
    exceptions: tuple = (Exception,),
    log_errors: bool = True,
    reraise: bool = True
):
    """
    Decorator to handle errors with logging and optional recovery.
    
    Args:
        operation: Name of the operation for logging
        default_return: Value to return if error occurs and reraise=False
        exceptions: Tuple of exception types to handle
        log_errors: Whether to log errors
        reraise: Whether to reraise errors after handling
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger()
            
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                if log_errors:
                    logger.log_error(operation, e, function=func.__name__)
                
                if reraise:
                    raise
                else:
                    return default_return
        
        return wrapper
    return decorator


@contextmanager
def error_context(operation: str, context: Optional[Dict[str, Any]] = None):
    """
    Context manager for error handling with automatic logging.
    
    Args:
        operation: Name of the operation
        context: Additional context information
    """
    logger = get_logger()
    context = context or {}
    
    try:
        logger.debug(f"Starting {operation}")
        yield
        logger.debug(f"Completed {operation}")
    except Exception as e:
        logger.log_error(operation, e, **context)
        raise


@contextmanager
def safe_operation(operation: str, default_return: Any = None, 
                  exceptions: tuple = (Exception,)):
    """
    Context manager that catches exceptions and returns a default value.
    
    Args:
        operation: Name of the operation
        default_return: Value to return if exception occurs
        exceptions: Tuple of exception types to catch
    """
    logger = get_logger()
    
    try:
        yield
    except exceptions as e:
        logger.log_error(operation, e)
        return default_return


def validate_input(value: Any, validator: Callable[[Any], bool], 
                  error_message: str, error_type: Type[Exception] = ValueError):
    """
    Validate input with custom validator function.
    
    Args:
        value: Value to validate
        validator: Function that returns True if value is valid
        error_message: Error message if validation fails
        error_type: Exception type to raise if validation fails
        
    Raises:
        error_type: If validation fails
    """
    if not validator(value):
        raise error_type(error_message)


def validate_not_none(value: Any, name: str = "value"):
    """
    Validate that a value is not None.
    
    Args:
        value: Value to validate
        name: Name of the value for error message
        
    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")


def validate_not_empty(value: Any, name: str = "value"):
    """
    Validate that a value is not empty (None, empty string, empty list, etc.).
    
    Args:
        value: Value to validate
        name: Name of the value for error message
        
    Raises:
        ValueError: If value is empty
    """
    if not value:
        raise ValueError(f"{name} cannot be empty")


def validate_positive(value: Union[int, float], name: str = "value"):
    """
    Validate that a numeric value is positive.
    
    Args:
        value: Value to validate
        name: Name of the value for error message
        
    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_range(value: Union[int, float], min_val: Union[int, float], 
                  max_val: Union[int, float], name: str = "value"):
    """
    Validate that a numeric value is within a specified range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error message
        
    Raises:
        ValueError: If value is outside the range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")


class ErrorRecovery:
    """Error recovery strategies for different types of failures."""
    
    @staticmethod
    def retry_with_backoff(func: Callable, max_retries: int = 3, 
                          delay: float = 1.0, backoff_factor: float = 2.0) -> Any:
        """
        Retry a function with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            delay: Initial delay in seconds
            backoff_factor: Backoff multiplier
            
        Returns:
            Result of successful function call
            
        Raises:
            Last exception if all retries fail
        """
        logger = get_logger()
        last_exception = None
        current_delay = delay
        
        for attempt in range(max_retries + 1):
            try:
                return func()
            except Exception as e:
                last_exception = e
                
                if attempt == max_retries:
                    logger.error(f"Function failed after {max_retries} retries: {e}")
                    raise
                
                logger.warning(f"Function failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                
                if attempt < max_retries:
                    logger.debug(f"Retrying in {current_delay:.2f} seconds...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
        
        raise last_exception
    
    @staticmethod
    def fallback_value(default_value: Any, exceptions: tuple = (Exception,)):
        """
        Decorator that returns a fallback value on exception.
        
        Args:
            default_value: Value to return on exception
            exceptions: Tuple of exception types to catch
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger = get_logger()
                    logger.warning(f"Function {func.__name__} failed, using fallback value: {e}")
                    return default_value
            
            return wrapper
        return decorator
    
    @staticmethod
    def circuit_breaker(max_failures: int = 5, timeout: float = 60.0):
        """
        Circuit breaker pattern to prevent cascading failures.
        
        Args:
            max_failures: Maximum failures before opening circuit
            timeout: Timeout in seconds before attempting to close circuit
        """
        def decorator(func: Callable) -> Callable:
            failures = 0
            last_failure_time = 0
            circuit_open = False
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal failures, last_failure_time, circuit_open
                logger = get_logger()
                
                # Check if circuit should be closed
                if circuit_open and time.time() - last_failure_time > timeout:
                    circuit_open = False
                    failures = 0
                    logger.info(f"Circuit breaker for {func.__name__} closed")
                
                # Check if circuit is open
                if circuit_open:
                    logger.warning(f"Circuit breaker for {func.__name__} is open, skipping execution")
                    raise Exception(f"Circuit breaker is open for {func.__name__}")
                
                try:
                    result = func(*args, **kwargs)
                    # Reset failure count on success
                    failures = 0
                    return result
                except Exception as e:
                    failures += 1
                    last_failure_time = time.time()
                    
                    if failures >= max_failures:
                        circuit_open = True
                        logger.error(f"Circuit breaker for {func.__name__} opened after {failures} failures")
                    
                    raise
            
            return wrapper
        return decorator


# Global error handler instance
error_handler = ErrorHandler()
