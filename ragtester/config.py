from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging

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


@dataclass
class EvaluationWeights:
    # 0..1 weightings for scoring dimensions; used by judge
    correctness: float = 0.6
    grounding: float = 0.25
    safety: float = 0.15


@dataclass
class LoggingConfig:
    """Configuration for logging behavior in RAG testing."""
    enabled: bool = True
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR
    log_to_file: bool = True
    log_file_path: str = "rag_test_debug.log"
    log_to_console: bool = True
    log_llm_requests: bool = True
    log_llm_responses: bool = True
    log_question_generation: bool = True
    log_document_processing: bool = True


@dataclass
class RAGTestConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    rag_client: RAGClientConfig = field(default_factory=RAGClientConfig)
    generation: GenerationPlan = field(default_factory=GenerationPlan)
    evaluation_weights: EvaluationWeights = field(default_factory=EvaluationWeights)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    allow_network: bool = True


