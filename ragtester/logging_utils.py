"""
Logging utilities for RAG testing framework.
Provides centralized logging configuration and utilities.
"""

import logging
import sys
from typing import Optional
from .config import LoggingConfig


class RAGLogger:
    """Centralized logger for RAG testing framework."""
    
    _instance: Optional['RAGLogger'] = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.logger = logging.getLogger("ragtester")
            self.config: Optional[LoggingConfig] = None
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
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Add console handler if enabled
        if config.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if enabled
        if config.log_to_file:
            try:
                file_handler = logging.FileHandler(config.log_file_path, mode='w')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                self.logger.warning(f"Could not create log file {config.log_file_path}: {e}")
        
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
                self.info(f"✅ Generated question for {metric}: '{question}'")
            else:
                self.warning(f"❌ Failed to generate question for {metric}")
                self.debug(f"Context used: {context[:200]}...")
    
    def log_document_processing(self, action: str, document_path: str, details: str = ""):
        """Log document processing details."""
        if self.config and self.config.log_document_processing:
            self.info(f"Document {action}: {document_path}")
            if details:
                self.debug(f"Details: {details}")


# Global logger instance
rag_logger = RAGLogger()


def get_logger() -> RAGLogger:
    """Get the global RAG logger instance."""
    return rag_logger


def configure_logging(config: LoggingConfig):
    """Configure the global RAG logger."""
    rag_logger.configure(config)
