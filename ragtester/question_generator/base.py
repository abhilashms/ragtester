from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

from ..types import Document, Question


class QuestionGenerator(ABC):
    @abstractmethod
    def generate(self, document_paths: List[str], num_per_category: dict, seed: int, num_pages: int = 5) -> List[Question]:
        raise NotImplementedError


