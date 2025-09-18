from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable, List

from ..types import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, paths: Iterable[str]) -> List[Document]:
        raise NotImplementedError


def make_document_id(path: str) -> str:
    """Generate a stable id from a path."""
    return os.path.normpath(path).replace("\\", "/")


