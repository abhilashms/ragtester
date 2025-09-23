from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..types import Evaluation, Question, RAGResponse


class ResponseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, responses: List[RAGResponse]) -> List[Evaluation]:
        raise NotImplementedError


