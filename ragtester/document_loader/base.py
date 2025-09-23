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
    - Preserving absolute paths from other platforms
    
    Args:
        path: File path (can be relative or absolute)
        
    Returns:
        Normalized absolute path with forward slashes
    """
    # Handle empty or None paths
    if not path:
        return ""
    
    # Check if this is already an absolute path
    # On Windows, absolute paths start with drive letter (C:) or UNC (\\)
    # On Unix-like systems, absolute paths start with /
    is_absolute = False
    
    # Check for Unix-style absolute paths (starting with /)
    if path.startswith('/'):
        is_absolute = True
        # For Unix paths, just normalize separators
        normalized = path.replace("\\", "/")
        return normalized
    
    # Check for Windows-style absolute paths
    if os.path.isabs(path):
        is_absolute = True
    
    if is_absolute:
        # For absolute paths, just normalize separators
        normalized = path.replace("\\", "/")
        return normalized
    else:
        # For relative paths, convert to absolute and normalize
        abs_path = os.path.abspath(path)
        normalized = abs_path.replace("\\", "/")
        return normalized


def make_document_id(path: str) -> str:
    """Generate a stable id from a path."""
    # Use normalized path for consistent document IDs across platforms
    return normalize_path(path)


