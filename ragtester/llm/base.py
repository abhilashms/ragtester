from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from ..types import LLMMessage


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: Sequence[LLMMessage], **kwargs: Any) -> str:
        raise NotImplementedError


