from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Iterable, List

from ..types import Document


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, paths: Iterable[str]) -> List[Document]:
        raise NotImplementedError


def normalize_path(path: str) -> str:
    """
    Normalize a file path for cross-platform compatibility.
    
    This function handles:
    - Converting relative paths to absolute paths
    - Normalizing path separators to forward slashes
    - Resolving parent directory references (..)
    - Resolving current directory references (.)
    - Handling both Windows and Linux path formats
    
    Args:
        path: File path (can be relative or absolute)
        
    Returns:
        Normalized absolute path with forward slashes
    """
    # Handle empty or None paths
    if not path:
        return ""
    
    # Convert to absolute path (this resolves relative paths, .., and .)
    abs_path = os.path.abspath(path)
    
    # Normalize path separators to forward slashes for consistency
    normalized = abs_path.replace("\\", "/")
    
    return normalized


def make_document_id(path: str) -> str:
    """Generate a stable id from a path."""
    # Use normalized path for consistent document IDs across platforms
    return normalize_path(path)


