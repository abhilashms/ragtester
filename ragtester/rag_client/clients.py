from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import urllib.request

from .base import RAGClient
from ..types import Document, RAGResponse, Question
from ..logging_utils import get_logger, log_operation
from ..exceptions import (
    RAGClientError, RAGClientConnectionError, RAGClientTimeoutError, 
    RAGClientResponseError, ValidationError
)
from ..error_handling import validate_not_empty, validate_not_none


class RESTRAGClient(RAGClient):
    def __init__(self, api_url: str, timeout_s: float = 30.0, extra_headers: Optional[Dict[str, str]] = None) -> None:
        # Validate inputs
        validate_not_empty(api_url, "API URL")
        validate_not_none(timeout_s, "timeout")
        
        if timeout_s <= 0:
            raise ValidationError(f"Timeout must be positive, got {timeout_s}")
        
        self.api_url = api_url.rstrip("/")
        self.timeout_s = timeout_s
        self.extra_headers = extra_headers or {}
        self.logger = get_logger()
        
        self.logger.debug(f"RESTRAGClient initialized: url={self.api_url}, timeout={timeout_s}s")

    def ask(self, query: str) -> RAGResponse:
        # Validate input
        validate_not_empty(query, "query")
        
        with log_operation("rag_rest_query", client="RESTRAGClient", query_length=len(query)):
            start_time = time.time()
            
            try:
                payload = json.dumps({"query": query}).encode("utf-8")
                req = urllib.request.Request(
                    self.api_url,
                    data=payload,
                    headers={"Content-Type": "application/json", **self.extra_headers},
                )
                
                self.logger.debug(f"Sending REST request to {self.api_url}")
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                    obj = json.loads(body)
                    
                response_time = time.time() - start_time
                self.logger.log_rag_operation(
                    "rest_query_success", 
                    query, 
                    response_time=response_time,
                    status_code=resp.status if hasattr(resp, 'status') else 'unknown'
                )
                
            except urllib.error.URLError as e:
                response_time = time.time() - start_time
                error = RAGClientConnectionError(
                    f"Failed to connect to RAG API: {e}",
                    error_code="RAG_CONNECTION_FAILED",
                    context={"url": self.api_url, "query_length": len(query)}
                )
                self.logger.log_error("rag_rest_connection", error, query=query[:100], response_time=response_time)
                raise error
            except urllib.error.HTTPError as e:
                response_time = time.time() - start_time
                error = RAGClientResponseError(
                    f"HTTP error from RAG API: {e.code} {e.reason}",
                    error_code="RAG_HTTP_ERROR",
                    context={"url": self.api_url, "status_code": e.code, "query_length": len(query)}
                )
                self.logger.log_error("rag_rest_http", error, query=query[:100], response_time=response_time)
                raise error
            except TimeoutError as e:
                response_time = time.time() - start_time
                error = RAGClientTimeoutError(
                    f"RAG API request timed out after {self.timeout_s}s",
                    error_code="RAG_TIMEOUT",
                    context={"url": self.api_url, "timeout": self.timeout_s, "query_length": len(query)}
                )
                self.logger.log_error("rag_rest_timeout", error, query=query[:100], response_time=response_time)
                raise error
            except json.JSONDecodeError as e:
                response_time = time.time() - start_time
                error = RAGClientResponseError(
                    f"Invalid JSON response from RAG API: {e}",
                    error_code="RAG_JSON_ERROR",
                    context={"url": self.api_url, "query_length": len(query)}
                )
                self.logger.log_error("rag_rest_json", error, query=query[:100], response_time=response_time)
                raise error
            except Exception as e:
                response_time = time.time() - start_time
                error = RAGClientError(
                    f"Unexpected error in RAG API request: {e}",
                    error_code="RAG_UNEXPECTED_ERROR",
                    context={"url": self.api_url, "query_length": len(query)}
                )
                self.logger.log_error("rag_rest_unexpected", error, query=query[:100], response_time=response_time)
                raise error
            
            answer = str(obj.get("answer", ""))
            context_docs: List[Document] = []
            context_errors = 0
            
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
                except Exception as e:
                    context_errors += 1
                    self.logger.debug(f"Error parsing context document: {e}")
                    continue
            
            if context_errors > 0:
                self.logger.warning(f"Failed to parse {context_errors} context documents")
            
            # Question is unknown here; caller should fill it when constructing
            dummy_question = Question(text=query, category=None)  # type: ignore[arg-type]
            
            self.logger.log_rag_operation(
                "rest_response_processed", 
                query, 
                response_time=response_time,
                context_docs=len(context_docs),
                answer_length=len(answer)
            )
            
            return RAGResponse(question=dummy_question, answer=answer, context_documents=context_docs)


class CallableRAGClient(RAGClient):
    def __init__(self, func) -> None:
        validate_not_none(func, "RAG function")
        self.func = func
        self.logger = get_logger()
        self.logger.debug(f"CallableRAGClient initialized with function: {func.__name__ if hasattr(func, '__name__') else 'anonymous'}")

    def ask(self, query: str) -> RAGResponse:
        # Validate input
        validate_not_empty(query, "query")
        
        with log_operation("rag_callable_query", client="CallableRAGClient", query_length=len(query)):
            start_time = time.time()
            
            try:
                self.logger.debug(f"Calling RAG function with query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
                result = self.func(query)
                
                response_time = time.time() - start_time
                self.logger.log_rag_operation(
                    "callable_query_success", 
                    query, 
                    response_time=response_time,
                    result_type=type(result).__name__
                )
                
            except Exception as e:
                response_time = time.time() - start_time
                error = RAGClientError(
                    f"Callable RAG function failed: {e}",
                    error_code="RAG_CALLABLE_ERROR",
                    context={"function_name": getattr(self.func, '__name__', 'anonymous'), "query_length": len(query)}
                )
                self.logger.log_error("rag_callable_query", error, query=query[:100], response_time=response_time)
                result = ""
            
            answer = str(result)
            dummy_question = Question(text=query, category=None)  # type: ignore[arg-type]
            
            self.logger.log_rag_operation(
                "callable_response_processed", 
                query, 
                response_time=response_time,
                answer_length=len(answer)
            )
            
            return RAGResponse(question=dummy_question, answer=answer, context_documents=[])


