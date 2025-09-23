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
from .logging_utils import configure_logging
from .exceptions import (
    RAGTesterError, DocumentProcessingError, RAGClientError, 
    QuestionGenerationError, EvaluationError, ValidationError
)
from .error_handling import (
    retry_on_error, handle_errors, error_context, 
    validate_not_none, validate_not_empty
)


class RAGTester:
    def __init__(self, rag_api_url: Optional[str] = None, rag_callable: Optional[callable] = None, config: Optional[RAGTestConfig] = None) -> None:
        self.config = config or RAGTestConfig()
        
        # Configure logging first
        configure_logging(self.config.logging)
        
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
        from .logging_utils import get_logger, log_operation
        
        logger = get_logger()
        
        # Validate input
        try:
            paths_list = list(paths)
            validate_not_empty(paths_list, "document paths")
        except Exception as e:
            raise ValidationError(f"Invalid document paths: {e}")
        
        with log_operation("document_upload", total_paths=len(paths_list)):
            # Only show document upload progress in debug mode
            logger.debug(f"📤 Uploading documents from {len(paths_list)} paths")
            
            try:
                docs = load_documents(paths_list)
                validate_not_empty(docs, "loaded documents")
                
                self.documents = docs
                self.document_paths = paths_list
                
                # Show document loading success as info (important for users)
                logger.info(f"✅ Loaded {len(self.documents)} documents from {len(self.document_paths)} paths")
                
                # Pass to client if it supports it
                try:
                    self.rag_client.set_corpus(self.documents)  # type: ignore[attr-defined]
                    logger.debug("Documents passed to RAG client successfully")
                except AttributeError:
                    logger.debug("RAG client does not support set_corpus method")
                except Exception as e:
                    logger.warning(f"Failed to pass documents to RAG client: {e}")
                
                return len(self.documents)
                
            except Exception as e:
                raise DocumentProcessingError(
                    f"Failed to upload documents: {e}",
                    error_code="DOC_UPLOAD_FAILED",
                    context={"paths_count": len(paths_list), "paths": paths_list[:5]}  # Log first 5 paths
                )

    def _ask_all(self, questions: List[Question]) -> List[RAGResponse]:
        from .logging_utils import get_logger, log_operation
        
        logger = get_logger()
        
        # Validate input
        validate_not_empty(questions, "questions list")
        
        with log_operation("rag_query_batch", total_questions=len(questions)):
            # Only show RAG query processing in debug mode
            logger.debug(f"🔍 Processing {len(questions)} RAG queries")
            
            responses: List[RAGResponse] = []
            successful_queries = 0
            failed_queries = 0
            
            for i, q in enumerate(questions):
                try:
                    validate_not_none(q, f"question {i}")
                    validate_not_empty(q.text, f"question {i} text")
                    
                    logger.debug(f"Processing question {i+1}/{len(questions)}: {q.text[:100]}{'...' if len(q.text) > 100 else ''}")
                    r = self.rag_client.ask(q.text)
                    # attach original question
                    r.question = q
                    responses.append(r)
                    successful_queries += 1
                except ValidationError as e:
                    logger.log_error("rag_query_validation", e, question_index=i)
                    failed_queries += 1
                    # Create a dummy response for validation failures
                    dummy_response = RAGResponse(question=q, answer="", context_documents=[])
                    responses.append(dummy_response)
                except RAGClientError as e:
                    logger.log_error("rag_query_client", e, question_index=i, question=q.text[:100])
                    failed_queries += 1
                    # Create a dummy response for client failures
                    dummy_response = RAGResponse(question=q, answer="", context_documents=[])
                    responses.append(dummy_response)
                except Exception as e:
                    logger.log_error("rag_query_unexpected", e, question_index=i, question=q.text[:100])
                    failed_queries += 1
                    # Create a dummy response for unexpected failures
                    dummy_response = RAGResponse(question=q, answer="", context_documents=[])
                    responses.append(dummy_response)
            
            # Show RAG query completion as info (important for users)
            logger.info(f"✅ RAG queries completed: {successful_queries} successful, {failed_queries} failed")
            
            if failed_queries > len(questions) * 0.5:  # More than 50% failed
                logger.warning(f"High failure rate: {failed_queries}/{len(questions)} queries failed")
            
            return responses

    def run_all_tests(self) -> TestResults:
        from .logging_utils import get_logger, log_operation
        
        logger = get_logger()
        
        with log_operation("full_test_run", categories=len(self.config.generation.per_category)):
            # Show important test run milestones
            logger.info("🚀 Starting full RAG test run")
            
            # Generate questions per category
            num_per_category = self.config.generation.per_category
            logger.info(f"❓ Generating questions for {len(num_per_category)} test categories")
            
            questions = self.question_generator.generate(self.document_paths, num_per_category, self.config.seed)
            logger.info(f"✅ Generated {len(questions)} questions total")
            
            responses = self._ask_all(questions)
            logger.info(f"✅ Collected {len(responses)} RAG responses")
            
            logger.info("⚖️ Starting evaluation phase")
            evaluations = self.judge.evaluate(responses)
            logger.info(f"✅ Completed evaluation of {len(evaluations)} responses")

            # Aggregate by category while preserving the order of questions within each category
            by_cat: Dict[TestCategory, List[Evaluation]] = defaultdict(list)
            for ev in evaluations:
                by_cat[ev.question.category].append(ev)
            
            # Create scorecards in the same order as the categories appear in the config
            # This ensures that questions from the same category appear consecutively
            scorecards = {}
            for category in self.config.generation.per_category.keys():
                if category in by_cat:
                    scorecards[category] = CategoryScorecard(category=category, evaluations=by_cat[category])
                    logger.debug(f"Created scorecard for {category.value}: {len(by_cat[category])} evaluations")
            
            results = TestResults(scorecards=scorecards)
            overall_score = results.overall_score()
            logger.info(f"🎉 Test run completed successfully! Overall score: {overall_score:.3f}")
            
            return results

    def export_results(self, results: TestResults, path: str) -> None:
        from .logging_utils import get_logger, log_operation
        
        logger = get_logger()
        
        with log_operation("results_export", output_path=path):
            logger.info(f"Exporting results to: {path}")
            
            try:
                if path.lower().endswith(".json"):
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(to_json(results))
                    logger.info("Results exported to JSON format")
                    return
                if path.lower().endswith(".csv"):
                    export_csv(results, path)
                    logger.info("Results exported to CSV format")
                    return
                if path.lower().endswith(".md"):
                    export_markdown(results, path)
                    logger.info("Results exported to Markdown format")
                    return
                if path.lower().endswith(".html"):
                    export_html(results, path)
                    logger.info("Results exported to HTML format")
                    return
                # default JSON
                with open(path, "w", encoding="utf-8") as f:
                    f.write(to_json(results))
                logger.info("Results exported to JSON format (default)")
                
            except Exception as e:
                logger.log_error("results_export", e, output_path=path)
                raise

    def print_summary(self, results: TestResults) -> None:
        from .logging_utils import get_logger
        
        logger = get_logger()
        
        with logger.operation_context("print_summary"):
            summary = to_console(results)
            print(summary)
            logger.info("Test summary printed to console")


