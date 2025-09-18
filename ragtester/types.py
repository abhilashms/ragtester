from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Protocol, Sequence, Tuple, TypedDict


class TestCategory(str, Enum):
    FAITHFULNESS = "Faithfulness"
    ANSWER_QUALITY = "Answer quality"
    TOXICITY = "Toxicity"
    ROBUSTNESS_RELIABILITY = "Robustness & reliability"
    SECURITY_SAFETY = "Security & safety"


class EvaluationMetric(str, Enum):
    FAITHFULNESS = "Faithfulness"
    ANSWER_QUALITY = "Answer quality"
    TOXICITY = "Toxicity"
    ROBUSTNESS_RELIABILITY = "Robustness & reliability"
    SECURITY_SAFETY = "Security & safety"


class Document(TypedDict):
    id: str
    path: str
    text: str
    meta: Dict[str, Any]


class LLMMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class Question:
    text: str
    category: TestCategory
    expected_source_ids: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    context_used: Optional[str] = None  # Store the context that was used to generate this question


@dataclass
class RAGResponse:
    question: Question
    answer: str
    context_documents: List[Document] = field(default_factory=list)
    latency_ms: Optional[float] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class MetricEvaluation:
    metric: EvaluationMetric
    score: float
    justification: str


@dataclass
class Evaluation:
    question: Question
    answer: str
    score: float
    verdict: str
    reasoning: str
    flags: Dict[str, Any] = field(default_factory=dict)
    metric_evaluations: List[MetricEvaluation] = field(default_factory=list)


@dataclass
class CategoryScorecard:
    category: TestCategory
    evaluations: List[Evaluation]

    @property
    def average_score(self) -> float:
        if not self.evaluations:
            return 0.0
        return sum(ev.score for ev in self.evaluations) / len(self.evaluations)


@dataclass
class TestResults:
    scorecards: Dict[TestCategory, CategoryScorecard]

    def overall_score(self) -> float:
        if not self.scorecards:
            return 0.0
        return sum(sc.average_score for sc in self.scorecards.values()) / len(self.scorecards)


class RAGCallable(Protocol):
    def __call__(self, query: str) -> str:  # minimal callable signature
        ...


class SupportsAsk(Protocol):
    def ask(self, query: str) -> RAGResponse:
        ...


class SupportsChat(Protocol):
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        ...


