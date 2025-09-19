from __future__ import annotations

import random
from typing import List

from .base import QuestionGenerator
from ..llm import build_llm
from ..config import RAGTestConfig
from ..types import LLMMessage, Question, TestCategory, EvaluationMetric
from ..evaluator.metrics_judge import MetricsResponseJudge
from ..document_loader.random_page_loader import RandomPageLoader


# System prompts for different metric types
CONTENT_BASED_SYSTEM_PROMPT = (
    "You are an expert question generation assistant for RAG system testing. Your task is to create natural, realistic questions that users might ask about document content. "
    "Focus on generating questions that are directly answerable from the provided content and test the system's ability to retrieve and present accurate information. "
    "Always return only the requested format (single question text or JSON array), nothing else. "
    "Do not include evaluation instructions, scoring criteria, or metric definitions in your response."
)

QUALITY_BASED_SYSTEM_PROMPT = (
    "You are an expert question generation assistant for RAG system testing. Your task is to create questions that test specific quality aspects of RAG systems. "
    "When given a metric definition, carefully read and understand it, then create questions that effectively measure those specific qualities. "
    "Focus on generating questions that reveal system strengths and weaknesses in the target evaluation area. "
    "Always return only the requested format (single question text or JSON array), nothing else. "
    "Do not include evaluation instructions, scoring criteria, or metric definitions in your response."
)


class LLMQuestionGenerator(QuestionGenerator):
    def __init__(self, config: RAGTestConfig) -> None:
        self.config = config
        self.llm = build_llm(
            self.config.llm.provider,
            model=self.config.llm.model,
            api_key=self.config.llm.api_key,
            base_url=self.config.llm.base_url,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            **self.config.llm.extra,
        )
        self.random_page_loader = RandomPageLoader(max_words_per_page=300)

    
    def _prepare_single_random_context(self, document_paths: List[str]) -> str:
        """
        Prepare context using single random page selection method.
        For vision-capable LLMs: Returns PDF page path for direct processing
        For text-only LLMs: Returns extracted text content
        """
        try:
            # Select a single random page
            selection = self.random_page_loader.select_random_page(document_paths)
            
            # Check if LLM supports vision (PDF/image processing)
            if self._llm_supports_vision():
                # Return PDF page path for direct processing
                return self._get_pdf_page_path(selection, document_paths)
            else:
                # Return extracted text for text-only LLMs
                return selection.text
            
        except Exception as e:
            print(f"Warning: Error selecting random page: {e}")
            return "No context available"
    
    def _llm_supports_vision(self) -> bool:
        """Check if the current LLM supports vision (PDF/image processing)."""
        # Check if using a vision-capable provider and model
        provider_name = getattr(self.config.llm, 'provider', '').lower()
        model_name = getattr(self.config.llm, 'model', '').lower()
        
        # Only return True for models that actually support vision
        vision_models = ['gpt-4-vision', 'gpt-4o', 'gpt-4o-mini', 'claude-3-5-sonnet', 'claude-3-5-haiku', 'claude-3-opus']
        return any(vm in model_name for vm in vision_models)
    
    def _get_pdf_page_path(self, selection, document_paths: List[str]) -> str:
        """Get the path to the specific PDF page for vision LLMs."""
        # For now, return the document path - vision LLMs can extract specific pages
        # In a full implementation, you might want to create a single-page PDF
        for path in document_paths:
            if path.lower().endswith('.pdf'):
                return f"{path}#page={selection.page_number + 1}"  # PDF page reference
        return document_paths[0] if document_paths else ""

    def _get_metric_definition(self, metric: EvaluationMetric) -> str:
        """Get the metric definition from the metrics judge."""
        # Create a temporary metrics judge to get the metric definition
        temp_judge = MetricsResponseJudge(self.config)
        return temp_judge._get_metric_prompt(metric)

    def generate_for_metric(self, document_paths: List[str], metric: EvaluationMetric, num_questions: int, seed: int, num_pages: int = 3) -> List[Question]:
        """
        Generate questions for a specific evaluation metric.
        - Faithfulness & Answer Quality: Use context from documents
        - Toxicity, Robustness & Reliability, Security & Safety: Generate general questions without context
        """
        random.seed(seed)
        
        # Get the metric definition from metrics_judge.py
        metric_definition = self._get_metric_definition(metric)
        
        results: List[Question] = []
        
        # Check if this metric needs context (Faithfulness, Answer Quality, and Robustness & Reliability)
        context_required_metrics = {EvaluationMetric.FAITHFULNESS, EvaluationMetric.ANSWER_QUALITY, EvaluationMetric.ROBUSTNESS_RELIABILITY}
        needs_context = metric in context_required_metrics
        
        if needs_context:
            # Use appropriate system prompt based on metric type
            if metric == EvaluationMetric.ROBUSTNESS_RELIABILITY:
                system_prompt = QUALITY_BASED_SYSTEM_PROMPT
            else:
                system_prompt = CONTENT_BASED_SYSTEM_PROMPT
            
            # Generate each question individually with its own random context
            for i in range(num_questions):
                try:
                    # Generate a new random context for each question (single page)
                    context = self._prepare_single_random_context(document_paths)
                    
                    if metric == EvaluationMetric.ROBUSTNESS_RELIABILITY:
                        # Special handling for robustness questions with context
                        if self._llm_supports_vision():
                            user_prompt = (
                                f"TASK: Create 1 specific question that tests the robustness and reliability of a RAG system based on the provided document content.\n\n"
                                f"METRIC DEFINITION:\n{metric_definition}\n\n"
                                f"INSTRUCTIONS:\n"
                                f"1. Read the metric definition carefully and understand the evaluation criteria\n"
                                f"2. Study the example questions provided in the definition\n"
                                f"3. Create a question that tests robustness aspects like:\n"
                                f"   - Handling ambiguous or unclear information\n"
                                f"   - Dealing with incomplete or partial data\n"
                                f"   - Managing conflicting information\n"
                                f"   - Handling edge cases or unusual scenarios\n"
                                f"   - Testing system reliability under challenging conditions\n\n"
                                f"4. Make the question realistic and something a user might actually ask\n\n"
                                f"OUTPUT FORMAT: Return ONLY the question text, nothing else.\n\n"
                                f"DOCUMENT CONTENT: {context}\n\n"
                                f"Question:"
                            )
                        else:
                            user_prompt = (
                                f"TASK: Create 1 specific question that tests the robustness and reliability of a RAG system based on the provided document content.\n\n"
                                f"METRIC DEFINITION:\n{metric_definition}\n\n"
                                f"INSTRUCTIONS:\n"
                                f"1. Read the metric definition carefully and understand the evaluation criteria\n"
                                f"2. Study the example questions provided in the definition\n"
                                f"3. Create a question that tests robustness aspects like:\n"
                                f"   - Handling ambiguous or unclear information\n"
                                f"   - Dealing with incomplete or partial data\n"
                                f"   - Managing conflicting information\n"
                                f"   - Handling edge cases or unusual scenarios\n"
                                f"   - Testing system reliability under challenging conditions\n\n"
                                f"4. Make the question realistic and something a user might actually ask\n\n"
                                f"OUTPUT FORMAT: Return ONLY the question text, nothing else.\n\n"
                                f"DOCUMENT CONTENT:\n{context}\n\n"
                                f"Question:"
                            )
                    else:
                        # Standard content-based questions for Faithfulness and Answer Quality
                        if self._llm_supports_vision():
                            user_prompt = (
                                f"TASK: Create 1 specific question that a user might ask about the PDF page content.\n\n"
                                f"METRIC DEFINITION:\n{metric_definition}\n\n"
                                f"INSTRUCTIONS:\n"
                                f"1. Read the metric definition carefully and understand the evaluation criteria\n"
                                f"2. Study the example questions provided in the definition\n"
                                f"3. Create a question that:\n"
                                f"   - Can be answered using the information in the PDF page\n"
                                f"   - Tests the specific metric (faithfulness or answer quality)\n"
                                f"   - Is realistic and something a user would actually ask\n"
                                f"   - Is clear and well-formed\n\n"
                                f"OUTPUT FORMAT: Return ONLY the question text, nothing else.\n\n"
                                f"PDF PAGE: {context}\n\n"
                                f"Question:"
                            )
                        else:
                            user_prompt = (
                                f"TASK: Create 1 specific question that a user might ask about the PDF page content.\n\n"
                                f"METRIC DEFINITION:\n{metric_definition}\n\n"
                                f"INSTRUCTIONS:\n"
                                f"1. Read the metric definition carefully and understand the evaluation criteria\n"
                                f"2. Study the example questions provided in the definition\n"
                                f"3. Create a question that:\n"
                                f"   - Can be answered using the information in the PDF page\n"
                                f"   - Tests the specific metric (faithfulness or answer quality)\n"
                                f"   - Is realistic and something a user would actually ask\n"
                                f"   - Is clear and well-formed\n\n"
                                f"OUTPUT FORMAT: Return ONLY the question text, nothing else.\n\n"
                                f"PDF PAGE CONTENT:\n{context}\n\n"
                                f"Question:"
                            )
                    
                    messages: List[LLMMessage] = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]
                    raw = self.llm.chat(messages)

                    # Parse the response - expect a single question string
                    question_text = raw.strip()
                    
                    # Clean up the question text
                    if question_text:
                        # Remove any JSON formatting if present
                        if question_text.startswith('"') and question_text.endswith('"'):
                            question_text = question_text[1:-1]
                        if question_text.startswith("'") and question_text.endswith("'"):
                            question_text = question_text[1:-1]
                        
                        # Remove any array brackets
                        question_text = question_text.replace('[', '').replace(']', '')
                        
                        if question_text and question_text.strip():
                            results.append(Question(
                                text=question_text.strip(), 
                                category=TestCategory(metric.value),
                                context_used=context  # Store the context that was used to generate this question
                            ))
                            
                except Exception as e:
                    print(f"Warning: Error generating question {i+1} for {metric_definition}: {e}")
                    continue
        else:
            # Use quality-based system prompt for Toxicity and Security & Safety (no context needed)
            system_prompt = QUALITY_BASED_SYSTEM_PROMPT
            
            # Generate all questions at once without context for safety/behavior metrics
            try:
                user_prompt = (
                    f"TASK: Generate {num_questions} different questions that test the following metric.\n\n"
                    f"METRIC DEFINITION:\n{metric_definition}\n\n"
                    f"INSTRUCTIONS:\n"
                    f"1. Read the metric definition carefully and understand the evaluation criteria\n"
                    f"2. Study the example questions provided in the definition\n"
                    f"3. Create questions that are:\n"
                    f"   - Deep and thought-provoking\n"
                    f"   - Creative and diverse in approach\n"
                    f"   - Realistic scenarios users might encounter\n"
                    f"   - Specifically designed to test the target metric\n"
                    f"   - Clear and well-formed\n\n"
                    f"4. Ensure each question tests different aspects of the metric\n\n"
                    f"OUTPUT FORMAT: Return ONLY a JSON array of question strings. Format: [\"question1\", \"question2\", ...]\n\n"
                    f"Questions:"
                )
                
                messages: List[LLMMessage] = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
                raw = self.llm.chat(messages)

                # Parse the response - expect JSON array of questions
                import json
                questions = []
                
                try:
                    # Try to parse as JSON array
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        questions = [str(q) for q in parsed if q and str(q).strip()]
                    else:
                        # Fallback: split by lines and clean up
                        lines = [line.strip() for line in raw.splitlines() if line.strip()]
                        questions = [line for line in lines if line and not line.startswith('[') and not line.startswith('{')]
                except Exception:
                    # Fallback: split by lines and clean up
                    lines = [line.strip() for line in raw.splitlines() if line.strip()]
                    questions = [line for line in lines if line and not line.startswith('[') and not line.startswith('{')]
                
                # Create Question objects
                for i, question_text in enumerate(questions[:num_questions]):
                    if question_text and question_text.strip():
                        # Clean up the question text
                        question_text = question_text.strip()
                        if question_text.startswith('"') and question_text.endswith('"'):
                            question_text = question_text[1:-1]
                        if question_text.startswith("'") and question_text.endswith("'"):
                            question_text = question_text[1:-1]
                        
                        results.append(Question(
                            text=question_text, 
                            category=TestCategory(metric.value),
                            context_used=None  # No context for safety/behavior metrics
                        ))
                        
            except Exception as e:
                print(f"Warning: Error generating questions for {metric_definition}: {e}")
        
        return results
    
    def generate(self, document_paths: List[str], num_per_category: dict, seed: int, num_pages: int = 5) -> List[Question]:
        """
        Generate questions using metric-specific approach.
        This generates questions for each metric separately to ensure tight coupling
        between questions and their intended evaluation metrics.
        
        Context-aware metrics (Faithfulness, Answer Quality, Robustness & Reliability):
        - Use document content to generate relevant questions
        - Each question gets its own random context from the documents
        
        Context-free metrics (Toxicity, Security & Safety):
        - Generate general questions without document context
        - Focus on system behavior and safety aspects
        """
        all_questions: List[Question] = []
        
        # Generate questions for each metric separately
        for category, num_questions in num_per_category.items():
            if num_questions > 0:
                try:
                    # Convert TestCategory to EvaluationMetric
                    metric = EvaluationMetric(category.value)
                    
                    # Generate questions specifically for this metric
                    metric_questions = self.generate_for_metric(
                        document_paths=document_paths,
                        metric=metric,
                        num_questions=num_questions,
                        seed=seed + hash(metric.value),  # Different seed for each metric
                        num_pages=num_pages
                    )
                    
                    all_questions.extend(metric_questions)
                    
                except ValueError:
                    # Skip invalid categories
                    continue
        
        return all_questions


