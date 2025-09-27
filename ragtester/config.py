from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import logging
import multiprocessing
import os

from .types import TestCategory


@dataclass
class LLMConfig:
    provider: str = "dummy"  # options: openai, anthropic, local, bedrock, dummy
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.2
    max_tokens: int = 512
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Automatically configure optimal threading for local models."""
        if self.provider == "local" and "n_threads" not in self.extra:
            self._set_optimal_threads()
    
    def _set_optimal_threads(self):
        """Set optimal number of threads based on system capabilities."""
        try:
            # Get CPU count
            cpu_count = multiprocessing.cpu_count()
            
            # Check for environment variable override
            env_threads = os.environ.get('RAGTESTER_N_THREADS')
            if env_threads:
                try:
                    n_threads = int(env_threads)
                    if n_threads > 0:
                        self.extra['n_threads'] = n_threads
                        return
                except ValueError:
                    pass
            
            # Optimal thread calculation:
            # - Use 75% of available cores for generation (single token)
            # - Use all available cores for batch processing (multiple tokens)
            # - Minimum of 2 threads, maximum of 16 for generation
            n_threads = max(2, min(16, int(cpu_count * 0.75)))
            n_threads_batch = max(2, cpu_count)
            
            self.extra['n_threads'] = n_threads
            self.extra['n_threads_batch'] = n_threads_batch
            
            # Log the configuration
            logging.info(f"Auto-configured threading: n_threads={n_threads}, n_threads_batch={n_threads_batch} (CPU cores: {cpu_count})")
            
        except Exception as e:
            # Fallback to safe defaults
            self.extra['n_threads'] = 4
            self.extra['n_threads_batch'] = 4
            logging.warning(f"Failed to auto-detect optimal threads, using defaults: {e}")


@dataclass
class RAGClientConfig:
    # Either URL for REST or leave None and use callable client
    api_url: Optional[str] = None
    timeout_s: float = 30.0
    extra_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class GenerationPlan:
    # Number of questions per category
    per_category: Dict[TestCategory, int] = field(
        default_factory=lambda: {
            TestCategory.FAITHFULNESS: 8,
            TestCategory.ANSWER_QUALITY: 8,
            TestCategory.TOXICITY: 6,
            TestCategory.ROBUSTNESS_RELIABILITY: 6,
            TestCategory.SECURITY_SAFETY: 6,
        }
    )
    # Question refinement settings
    enable_refinement: bool = True  # Enable rule-based question refinement by default


@dataclass
class EvaluationWeights:
    # 0..1 weightings for scoring dimensions; used by judge
    correctness: float = 0.6
    grounding: float = 0.25
    safety: float = 0.15


@dataclass
class LoggingConfig:
    """
    Configuration for logging behavior in RAG testing.
    
    Log Levels:
    - DEBUG: All messages (for development/debugging)
    - INFO: Important progress steps (default - recommended for users)
    - WARNING: Warnings and potential issues
    - ERROR: Errors only
    
    Examples:
        # See all important progress (recommended)
        config = LoggingConfig(level="INFO")
        
        # Quiet mode - only warnings and errors
        config = LoggingConfig(level="WARNING")
        
        # Verbose mode - see everything
        config = LoggingConfig(level="DEBUG")
        
        # Disable console output, only log to file
        config = LoggingConfig(log_to_console=False, log_to_file=True)
    """
    enabled: bool = True
    level: str = "INFO"  # Default to INFO so users can see important progress
    log_to_file: bool = True
    log_file_path: str = "rag_test_debug.log"
    log_to_console: bool = True
    log_llm_requests: bool = False  # Reduced verbosity for users
    log_llm_responses: bool = False  # Reduced verbosity for users
    log_question_generation: bool = False  # Reduced verbosity - only show important progress
    log_document_processing: bool = False  # Reduced verbosity - only show important progress
    log_rag_operations: bool = False  # Reduced verbosity - only show important progress
    log_evaluation_operations: bool = False  # Reduced verbosity - only show important progress
    log_performance_metrics: bool = False  # Reduced verbosity - only show important progress
    log_errors: bool = True  # Always show errors
    max_log_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    structured_logging: bool = True
    include_timestamps: bool = True
    include_thread_info: bool = False
    include_process_info: bool = False
    # New option for verbose mode
    verbose_mode: bool = False  # Set to True to see more detailed progress


@dataclass
class RAGTestConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag_client: RAGClientConfig = field(default_factory=RAGClientConfig)
    generation: GenerationPlan = field(default_factory=GenerationPlan)
    evaluation_weights: EvaluationWeights = field(default_factory=EvaluationWeights)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    allow_network: bool = True


