from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import urllib.request

from .base import RAGClient
from ..types import Document, RAGResponse, Question


class RESTRAGClient(RAGClient):
    def __init__(self, api_url: str, timeout_s: float = 30.0, extra_headers: Optional[Dict[str, str]] = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}

    def ask(self, query: str) -> RAGResponse:
        payload = json.dumps({"query": query}).encode("utf-8")
        req = urllib.request.Request(
            self.api_url,
            data=payload,
            headers={"Content-Type": "application/json", **self.extra_headers},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8")
                obj = json.loads(body)
        except Exception:
            obj = {"answer": "", "context": []}
        answer = str(obj.get("answer", ""))
        context_docs: List[Document] = []
        for c in obj.get("context", []) or []:
            try:
                context_docs.append(
                    Document(
                        id=str(c.get("id", "")),
                        path=str(c.get("path", "")),
                        text=str(c.get("text", "")),
                        meta={k: v for k, v in c.items() if k not in ("id", "path", "text")},
                    )
                )
            except Exception:
                continue
        # Question is unknown here; caller should fill it when constructing
        dummy_question = Question(text=query, category=None)  # type: ignore[arg-type]
        return RAGResponse(question=dummy_question, answer=answer, context_documents=context_docs)


class CallableRAGClient(RAGClient):
    def __init__(self, func) -> None:
        self.func = func

    def ask(self, query: str) -> RAGResponse:
        try:
            result = self.func(query)
        except Exception:
            result = ""
        answer = str(result)
        dummy_question = Question(text=query, category=None)  # type: ignore[arg-type]
        return RAGResponse(question=dummy_question, answer=answer, context_documents=[])


