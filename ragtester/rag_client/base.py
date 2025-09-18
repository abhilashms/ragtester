from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..types import Document, RAGResponse


class RAGClient(ABC):
    @abstractmethod
    def ask(self, query: str) -> RAGResponse:
        raise NotImplementedError

    def set_corpus(self, documents: List[Document]) -> None:
        # Optional: for local RAGs that need documents
        return None


