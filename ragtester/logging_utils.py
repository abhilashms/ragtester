"""
Logging utilities for RAG testing framework.
Provides centralized logging configuration and utilities.
"""

import logging
import logging.handlers
import sys
import time
import uuid
import threading
import os
import platform
from typing import Optional, Dict, Any
from contextlib import contextmanager
from .config import LoggingConfig
from .exceptions import RAGTesterError, wrap_exception


class UnicodeSafeFormatter(logging.Formatter):
    """Custom formatter that handles Unicode characters safely on Windows."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_windows = platform.system() == "Windows"
        self._console_encoding = self._get_console_encoding()
    
    def _get_console_encoding(self) -> str:
        """Get the console encoding, with fallback to UTF-8."""
        try:
            if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
                return sys.stdout.encoding
            elif hasattr(sys.stderr, 'encoding') and sys.stderr.encoding:
                return sys.stderr.encoding
            else:
                return 'utf-8'
        except:
            return 'utf-8'
    
    def _safe_encode(self, message: str) -> str:
        """Safely encode message for console output."""
        if not self._is_windows:
            return message
        
        # Check if the message can be encoded with the console encoding
        try:
            message.encode(self._console_encoding)
            return message
        except UnicodeEncodeError:
            # Replace problematic Unicode characters with safe alternatives
            return self._replace_unicode_chars(message)
    
    def _replace_unicode_chars(self, message: str) -> str:
        """Replace Unicode characters with safe alternatives for Windows console."""
        replacements = {
            '📤': '[UPLOAD]',
            '📄': '[DOC]',
            '🔍': '[SEARCH]',
            '⚖️': '[EVAL]',
            '⏱️': '[TIME]',
            '❌': '[ERROR]',
            '✅': '[OK]',
            '✓': '[OK]',
            '✗': '[FAIL]',
            '⚠️': '[WARN]',
            '🚀': '[START]',
            '🎉': '[SUCCESS]',
            '💡': '[INFO]',
            '🔧': '[CONFIG]',
            '📊': '[STATS]',
            '🏆': '[WIN]',
            '❤️': '[LOVE]',
            '📋': '[LIST]',
            '🔧': '[TOOL]',
            '📦': '[PACKAGE]',
            '🌐': '[WEB]',
            '💾': '[SAVE]',
            '📁': '[FOLDER]',
            '📝': '[NOTE]',
            '🔗': '[LINK]',
            '⭐': '[STAR]',
            '🎯': '[TARGET]',
            '🚨': '[ALERT]',
            '🔒': '[LOCK]',
            '🔓': '[UNLOCK]',
            '📈': '[UP]',
            '📉': '[DOWN]',
            '🔄': '[REFRESH]',
            '⏸️': '[PAUSE]',
            '▶️': '[PLAY]',
            '⏹️': '[STOP]',
            '⏭️': '[NEXT]',
            '⏮️': '[PREV]',
            '🔀': '[SHUFFLE]',
            '🔁': '[REPEAT]',
            '🔂': '[REPEAT_ONE]',
            '🔃': '[REFRESH_CCW]',
            '🔄': '[REFRESH_CW]',
            '🔅': '[DIM]',
            '🔆': '[BRIGHT]',
            '🔇': '[MUTE]',
            '🔈': '[VOL_LOW]',
            '🔉': '[VOL_MED]',
            '🔊': '[VOL_HIGH]',
            '🔋': '[BATTERY]',
            '🔌': '[PLUG]',
            '🔍': '[SEARCH]',
            '🔎': '[SEARCH_RIGHT]',
            '🔏': '[LOCK_INK]',
            '🔐': '[LOCK_KEY]',
            '🔑': '[KEY]',
            '🔒': '[LOCK]',
            '🔓': '[UNLOCK]',
            '🔔': '[BELL]',
            '🔕': '[BELL_SLASH]',
            '🔖': '[BOOKMARK]',
            '🔗': '[LINK]',
            '🔘': '[RADIO]',
            '🔙': '[BACK]',
            '🔚': '[END]',
            '🔛': '[ON]',
            '🔜': '[SOON]',
            '🔝': '[TOP]',
            '🔞': '[18+]',
            '🔟': '[10]',
            '🔠': '[ABC]',
            '🔡': '[abc]',
            '🔢': '[123]',
            '🔣': '[SYMBOLS]',
            '🔤': '[LETTERS]',
            '🔥': '[FIRE]',
            '🔦': '[FLASHLIGHT]',
            '🔧': '[WRENCH]',
            '🔨': '[HAMMER]',
            '🔩': '[NUT_BOLT]',
            '🔪': '[KNIFE]',
            '🔫': '[GUN]',
            '🔬': '[MICROSCOPE]',
            '🔭': '[TELESCOPE]',
            '🔮': '[CRYSTAL_BALL]',
            '🔯': '[STAR_DAVID]',
            '🔰': '[BEGINNER]',
            '🔱': '[TRIDENT]',
            '🔲': '[BLACK_SQUARE]',
            '🔳': '[WHITE_SQUARE]',
            '🔴': '[RED_CIRCLE]',
            '🔵': '[BLUE_CIRCLE]',
            '🔶': '[ORANGE_DIAMOND]',
            '🔷': '[BLUE_DIAMOND]',
            '🔸': '[ORANGE_DIAMOND_SMALL]',
            '🔹': '[BLUE_DIAMOND_SMALL]',
            '🔺': '[RED_TRIANGLE]',
            '🔻': '[RED_TRIANGLE_DOWN]',
            '🔼': '[TRIANGLE_UP]',
            '🔽': '[TRIANGLE_DOWN]',
            '🕐': '[CLOCK1]',
            '🕑': '[CLOCK2]',
            '🕒': '[CLOCK3]',
            '🕓': '[CLOCK4]',
            '🕔': '[CLOCK5]',
            '🕕': '[CLOCK6]',
            '🕖': '[CLOCK7]',
            '🕗': '[CLOCK8]',
            '🕘': '[CLOCK9]',
            '🕙': '[CLOCK10]',
            '🕚': '[CLOCK11]',
            '🕛': '[CLOCK12]',
            '🕜': '[CLOCK130]',
            '🕝': '[CLOCK230]',
            '🕞': '[CLOCK330]',
            '🕟': '[CLOCK430]',
            '🕠': '[CLOCK530]',
            '🕡': '[CLOCK630]',
            '🕢': '[CLOCK730]',
            '🕣': '[CLOCK830]',
            '🕤': '[CLOCK930]',
            '🕥': '[CLOCK1030]',
            '🕦': '[CLOCK1130]',
            '🕧': '[CLOCK1230]',
        }
        
        for unicode_char, replacement in replacements.items():
            message = message.replace(unicode_char, replacement)
        
        return message
    
    def format(self, record):
        """Format the log record with Unicode safety."""
        # Get the formatted message
        formatted = super().format(record)
        
        # Apply Unicode safety for console output
        if hasattr(record, 'stream') and record.stream in (sys.stdout, sys.stderr):
            formatted = self._safe_encode(formatted)
        
        return formatted


class UnicodeSafeStreamHandler(logging.StreamHandler):
    """Custom stream handler that handles Unicode characters safely."""
    
    def __init__(self, stream=None):
        super().__init__(stream)
        self._is_windows = platform.system() == "Windows"
        self._console_encoding = self._get_console_encoding()
    
    def _get_console_encoding(self) -> str:
        """Get the console encoding, with fallback to UTF-8."""
        try:
            if hasattr(self.stream, 'encoding') and self.stream.encoding:
                return self.stream.encoding
            else:
                return 'utf-8'
        except:
            return 'utf-8'
    
    def emit(self, record):
        """Emit a log record with Unicode safety."""
        try:
            msg = self.format(record)
            
            # For Windows console, ensure proper encoding
            if self._is_windows and hasattr(self.stream, 'encoding'):
                try:
                    # Try to encode with the stream's encoding
                    msg.encode(self.stream.encoding)
                except UnicodeEncodeError:
                    # If encoding fails, replace Unicode characters
                    formatter = UnicodeSafeFormatter()
                    msg = formatter._replace_unicode_chars(msg)
            
            # Write the message
            self.stream.write(msg + self.terminator)
            self.flush()
            
        except Exception:
            self.handleError(record)


class RAGLogger:
    """Centralized logger for RAG testing framework."""
    
    _instance: Optional['RAGLogger'] = None
    _initialized = False
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self.logger = logging.getLogger("ragtester")
                    self.config: Optional[LoggingConfig] = None
                    self._context = threading.local()
                    self._initialized = True
    
    def configure(self, config: LoggingConfig):
        """Configure the logger with the given configuration."""
        self.config = config
        
        if not config.enabled:
            self.logger.setLevel(logging.CRITICAL)
            return
        
        # Set log level
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        self.logger.setLevel(level_map.get(config.level, logging.INFO))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter with enhanced information
        format_parts = []
        if config.include_timestamps:
            format_parts.append("%(asctime)s")
        format_parts.append("[%(levelname)s]")
        format_parts.append("%(name)s")
        if config.include_thread_info:
            format_parts.append("[%(threadName)s]")
        if config.include_process_info:
            format_parts.append("[%(processName)s:%(process)d]")
        format_parts.append(": %(message)s")
        
        # Use Unicode-safe formatter
        formatter = UnicodeSafeFormatter(
            " ".join(format_parts),
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Add console handler if enabled
        if config.log_to_console:
            console_handler = UnicodeSafeStreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler with rotation if enabled
        if config.log_to_file:
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    config.log_file_path,
                    maxBytes=config.max_log_file_size,
                    backupCount=config.backup_count,
                    mode='a',
                    encoding='utf-8'  # Ensure UTF-8 encoding for file logging
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not create log file {config.log_file_path}: {e}")
        
        # Only log configuration success in debug mode to reduce verbosity
        if self.config.level == "DEBUG":
            self.logger.info("RAG Logger configured successfully")
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        if self.config and self.config.enabled:
            self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        if self.config and self.config.enabled:
            self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        if self.config and self.config.enabled:
            self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        if self.config and self.config.enabled:
            self.logger.error(message, **kwargs)
    
    def log_llm_request(self, provider: str, model: str, messages: list, **kwargs):
        """Log LLM request details."""
        if self.config and self.config.log_llm_requests:
            self.debug(f"LLM Request - Provider: {provider}, Model: {model}")
            self.debug(f"LLM Request - Messages: {messages}")
            if kwargs:
                self.debug(f"LLM Request - Extra params: {kwargs}")
    
    def log_llm_response(self, response: str, response_time: Optional[float] = None):
        """Log LLM response details."""
        if self.config and self.config.log_llm_responses:
            self.debug(f"LLM Response: '{response}'")
            if response_time:
                self.debug(f"LLM Response time: {response_time:.2f}s")
    
    def log_question_generation(self, metric: str, context: str, question: str, success: bool):
        """Log question generation details."""
        if self.config and self.config.log_question_generation:
            if success:
                self.info(f"✓ Generated question for {metric}")
            else:
                self.warning(f"✗ Failed to generate question for {metric}")
                self.debug(f"Context used: {context[:200]}...")
    
    def log_document_processing(self, action: str, document_path: str, details: str = ""):
        """Log document processing details."""
        if self.config and self.config.log_document_processing:
            self.info(f"📄 Document {action}: {document_path}")
            if details:
                self.debug(f"Details: {details}")
    
    def log_rag_operation(self, operation: str, query: str, response_time: Optional[float] = None, 
                         context_docs: int = 0, **kwargs):
        """Log RAG operation details."""
        if self.config and self.config.log_rag_operations:
            msg = f"🔍 RAG {operation}: '{query[:100]}{'...' if len(query) > 100 else ''}'"
            if response_time:
                msg += f" ({response_time:.2f}s)"
            if context_docs > 0:
                msg += f" [context: {context_docs} docs]"
            if kwargs:
                msg += f" {kwargs}"
            self.info(msg)
    
    def log_evaluation_operation(self, operation: str, metric: str, question: str, 
                                score: Optional[float] = None, **kwargs):
        """Log evaluation operation details."""
        if self.config and self.config.log_evaluation_operations:
            msg = f"⚖️ Evaluation {operation}: {metric}"
            if score is not None:
                msg += f" (score: {score:.2f})"
            msg += f" - '{question[:100]}{'...' if len(question) > 100 else ''}'"
            if kwargs:
                msg += f" {kwargs}"
            self.info(msg)
    
    def log_performance_metric(self, operation: str, duration: float, **kwargs):
        """Log performance metrics."""
        if self.config and self.config.log_performance_metrics:
            msg = f"⏱️ {operation} completed in {duration:.3f}s"
            if kwargs:
                msg += f" {kwargs}"
            self.info(msg)
    
    def log_error(self, operation: str, error: Exception, **kwargs):
        """Log error details with context."""
        if self.config and self.config.log_errors:
            # Enhanced error logging with more context
            error_type = type(error).__name__
            error_msg = str(error)
            
            # Add error code if it's a RAGTesterError
            if isinstance(error, RAGTesterError) and error.error_code:
                error_type = f"{error_type}[{error.error_code}]"
            
            msg = f"❌ Error in {operation}: {error_type}: {error_msg}"
            
            # Add context information
            if kwargs:
                context_parts = []
                for key, value in kwargs.items():
                    if isinstance(value, (str, int, float, bool)):
                        context_parts.append(f"{key}={value}")
                    else:
                        context_parts.append(f"{key}={type(value).__name__}")
                if context_parts:
                    msg += f" (Context: {', '.join(context_parts)})"
            
            # Add original error context if available
            if isinstance(error, RAGTesterError) and error.context:
                original_context = []
                for key, value in error.context.items():
                    if isinstance(value, (str, int, float, bool)):
                        original_context.append(f"{key}={value}")
                    else:
                        original_context.append(f"{key}={type(value).__name__}")
                if original_context:
                    msg += f" (Original context: {', '.join(original_context)})"
            
            self.error(msg)
            
            # Log original error if it's wrapped
            if isinstance(error, RAGTesterError) and error.original_error:
                self.debug(f"Original error: {type(error.original_error).__name__}: {error.original_error}")
    
    def log_structured_error(self, operation: str, error: Exception, 
                           severity: str = "ERROR", **kwargs):
        """Log error with structured information for better analysis."""
        if self.config and self.config.log_errors:
            error_info = {
                "operation": operation,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "severity": severity,
                "timestamp": time.time(),
                **kwargs
            }
            
            # Add RAGTesterError specific information
            if isinstance(error, RAGTesterError):
                if error.error_code:
                    error_info["error_code"] = error.error_code
                if error.context:
                    error_info["error_context"] = error.context
                if error.original_error:
                    error_info["original_error"] = {
                        "type": type(error.original_error).__name__,
                        "message": str(error.original_error)
                    }
            
            # Log as structured JSON if structured logging is enabled
            if self.config.structured_logging:
                import json
                self.error(f"STRUCTURED_ERROR: {json.dumps(error_info, default=str)}")
            else:
                # Fall back to regular error logging
                self.log_error(operation, error, **kwargs)
    
    def set_context(self, **context):
        """Set logging context for the current thread."""
        self._context.data = getattr(self._context, 'data', {})
        self._context.data.update(context)
    
    def get_context(self) -> Dict[str, Any]:
        """Get current logging context."""
        return getattr(self._context, 'data', {})
    
    def clear_context(self):
        """Clear logging context for the current thread."""
        self._context.data = {}
    
    @contextmanager
    def operation_context(self, operation: str, **context):
        """Context manager for logging operations with timing and enhanced error handling."""
        start_time = time.time()
        operation_id = str(uuid.uuid4())[:8]
        
        # Set context
        self.set_context(operation=operation, operation_id=operation_id, **context)
        
        try:
            self.info(f"Starting {operation} [ID: {operation_id}]")
            yield operation_id
        except Exception as e:
            duration = time.time() - start_time
            
            # Wrap generic exceptions in RAGTesterError for better handling
            if not isinstance(e, RAGTesterError):
                wrapped_error = wrap_exception(
                    e, 
                    f"Operation {operation} failed", 
                    error_code=f"OP_{operation.upper()}_FAILED",
                    context={"operation_id": operation_id, "duration": duration, **context}
                )
                self.log_structured_error(operation, wrapped_error, 
                                        operation_id=operation_id, duration=duration)
                raise wrapped_error
            else:
                self.log_structured_error(operation, e, 
                                        operation_id=operation_id, duration=duration)
                raise
        finally:
            duration = time.time() - start_time
            self.log_performance_metric(operation, duration, operation_id=operation_id)
            self.info(f"Completed {operation} [ID: {operation_id}] in {duration:.3f}s")
            self.clear_context()


# Global logger instance
rag_logger = RAGLogger()


def get_logger() -> RAGLogger:
    """Get the global RAG logger instance."""
    return rag_logger


def configure_logging(config: LoggingConfig):
    """Configure the global RAG logger."""
    rag_logger.configure(config)


def safe_unicode_message(message: str) -> str:
    """
    Convert a message with Unicode characters to a safe version for Windows console.
    
    This function provides a centralized way to handle Unicode characters
    that might cause encoding issues on Windows systems.
    
    Args:
        message: The message that may contain Unicode characters
        
    Returns:
        A Unicode-safe version of the message
    """
    if platform.system() != "Windows":
        return message
    
    # Check if the message can be encoded with the console encoding
    try:
        console_encoding = 'cp1252'  # Default Windows console encoding
        if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding:
            console_encoding = sys.stdout.encoding
        
        message.encode(console_encoding)
        return message
    except UnicodeEncodeError:
        # Replace problematic Unicode characters with safe alternatives
        formatter = UnicodeSafeFormatter()
        return formatter._replace_unicode_chars(message)


@contextmanager
def log_operation(operation: str, **context):
    """Context manager for logging operations with timing and error handling."""
    with rag_logger.operation_context(operation, **context) as operation_id:
        yield operation_id
