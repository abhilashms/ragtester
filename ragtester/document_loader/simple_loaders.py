from __future__ import annotations

import os
from typing import Iterable, List

from .base import DocumentLoader, make_document_id
from ..types import Document


class TextFileLoader(DocumentLoader):
    def load(self, paths: Iterable[str]) -> List[Document]:
        documents: List[Document] = []
        for path in paths:
            if not os.path.isfile(path):
                continue
            if not path.lower().endswith((".txt", ".md", ".rst")):
                continue
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                documents.append(
                    Document(id=make_document_id(path), path=path, text=text, meta={"type": "text"})
                )
            except Exception:
                # Skip unreadable files
                continue
        return documents


class PDFFileLoader(DocumentLoader):
    def __init__(self) -> None:
        try:
            import PyPDF2  # type: ignore

            self._pypdf2_available = True
        except Exception:
            self._pypdf2_available = False

    def load(self, paths: Iterable[str]) -> List[Document]:
        documents: List[Document] = []
        if not self._pypdf2_available:
            return documents

        import PyPDF2  # type: ignore

        for path in paths:
            if not os.path.isfile(path) or not path.lower().endswith(".pdf"):
                continue
            try:
                with open(path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    texts = []
                    for page in getattr(reader, "pages", []):
                        try:
                            texts.append(page.extract_text() or "")
                        except Exception:
                            continue
                    text = "\n".join(texts)
                documents.append(
                    Document(id=make_document_id(path), path=path, text=text, meta={"type": "pdf"})
                )
            except Exception:
                continue
        return documents


def load_documents(paths: Iterable[str]) -> List[Document]:
    loaders: List[DocumentLoader] = [TextFileLoader(), PDFFileLoader()]
    all_docs: List[Document] = []
    for loader in loaders:
        all_docs.extend(loader.load(paths))
    # Deduplicate by id
    seen = set()
    deduped: List[Document] = []
    for d in all_docs:
        if d["id"] in seen:
            continue
        seen.add(d["id"])
        deduped.append(d)
    return deduped


