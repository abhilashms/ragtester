from __future__ import annotations

import json
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional

from .config import RAGTestConfig
from .document_loader import load_documents
from .evaluator.metrics_judge import MetricsResponseJudge
from .question_generator.generators import LLMQuestionGenerator
from .rag_client import CallableRAGClient, RESTRAGClient, RAGClient
from .reporter import export_csv, export_html, export_markdown, to_console, to_json
from .types import CategoryScorecard, Evaluation, Question, RAGResponse, TestCategory, TestResults


class RAGTester:
    def __init__(self, rag_api_url: Optional[str] = None, rag_callable: Optional[callable] = None, config: Optional[RAGTestConfig] = None) -> None:
        self.config = config or RAGTestConfig()
        self.documents: List[dict] = []
        self.document_paths: List[str] = []
        # Prepare RAG client
        if rag_api_url:
            self.rag_client: RAGClient = RESTRAGClient(
                rag_api_url, timeout_s=self.config.rag_client.timeout_s, extra_headers=self.config.rag_client.extra_headers
            )
        elif rag_callable is not None:
            self.rag_client = CallableRAGClient(rag_callable)
        else:
            # Default to callable that echoes question; minimal offline behavior
            self.rag_client = CallableRAGClient(lambda q: f"Echo: {q}")

        self.question_generator = LLMQuestionGenerator(self.config)
        self.judge = MetricsResponseJudge(self.config)
        random.seed(self.config.seed)

    def upload_documents(self, paths: Iterable[str]) -> int:
        docs = load_documents(paths)
        self.documents = docs
        self.document_paths = list(paths)
        # Pass to client if it supports it
        try:
            self.rag_client.set_corpus(self.documents)  # type: ignore[attr-defined]
        except Exception:
            pass
        return len(self.documents)

    def _ask_all(self, questions: List[Question]) -> List[RAGResponse]:
        responses: List[RAGResponse] = []
        for q in questions:
            r = self.rag_client.ask(q.text)
            # attach original question
            r.question = q
            responses.append(r)
        return responses

    def run_all_tests(self) -> TestResults:
        # Generate questions per category
        num_per_category = self.config.generation.per_category
        questions = self.question_generator.generate(self.document_paths, num_per_category, self.config.seed)
        responses = self._ask_all(questions)
        evaluations = self.judge.evaluate(responses)

        # Aggregate by category
        by_cat: Dict[TestCategory, List[Evaluation]] = defaultdict(list)
        for ev in evaluations:
            by_cat[ev.question.category].append(ev)
        scorecards = {cat: CategoryScorecard(category=cat, evaluations=evs) for cat, evs in by_cat.items()}
        return TestResults(scorecards=scorecards)

    def export_results(self, results: TestResults, path: str) -> None:
        if path.lower().endswith(".json"):
            with open(path, "w", encoding="utf-8") as f:
                f.write(to_json(results))
            return
        if path.lower().endswith(".csv"):
            export_csv(results, path)
            return
        if path.lower().endswith(".md"):
            export_markdown(results, path)
            return
        if path.lower().endswith(".html"):
            export_html(results, path)
            return
        # default JSON
        with open(path, "w", encoding="utf-8") as f:
            f.write(to_json(results))

    def print_summary(self, results: TestResults) -> None:
        print(to_console(results))


