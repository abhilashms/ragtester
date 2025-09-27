from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from typing import Dict

from ..types import CategoryScorecard, TestCategory, TestResults
from ..logging_utils import get_logger, log_operation


def to_json(results: TestResults) -> str:
    logger = get_logger()
    
    with log_operation("json_export", format="json"):
        logger.debug("Converting test results to JSON format")
        
        data = {
            "overall_score": results.overall_score(),
            "scorecards": {
                cat.value: {
                    "average_score": sc.average_score,
                    "evaluations": [
                        {
                            "question": e.question.text,
                            "category": e.question.category.value,
                            "score": e.score if e.score is not None else None,  # Handle null scores
                            "verdict": e.verdict,
                            "reasoning": e.reasoning,
                            "page_number": e.question.page_number,
                        }
                        for e in sc.evaluations
                    ],
                }
                for cat, sc in results.scorecards.items()
            },
        }
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        logger.debug(f"JSON export completed: {len(json_str)} characters, {len(results.scorecards)} categories")
        return json_str


def to_console(results: TestResults) -> str:
    logger = get_logger()
    
    with log_operation("console_export", format="console"):
        logger.debug("Converting test results to console format")
        
        lines = []
        lines.append(f"Overall score: {results.overall_score():.3f}")
        for cat, sc in results.scorecards.items():
            lines.append(f"- {cat.value}: avg={sc.average_score:.3f} ({len(sc.evaluations)} items)")
        
        console_str = "\n".join(lines)
        logger.debug(f"Console export completed: {len(lines)} lines, {len(results.scorecards)} categories")
        return console_str


def export_csv(results: TestResults, path: str) -> None:
    logger = get_logger()
    
    with log_operation("csv_export", format="csv", output_path=path):
        try:
            logger.debug(f"Exporting results to CSV: {path}")
            
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["category", "question", "rag_answer", "score", "reasoning", "page_number"])
                
                total_rows = 0
                for cat, sc in results.scorecards.items():
                    for e in sc.evaluations:
                        page_num = e.question.page_number if e.question.page_number is not None else ""
                        score_val = e.score if e.score is not None else ""  # Handle null scores
                        writer.writerow([cat.value, e.question.text, e.answer, score_val, e.reasoning, page_num])
                        total_rows += 1
            
            file_size = os.path.getsize(path)
            logger.debug(f"CSV export completed: {total_rows} rows written to {path} ({file_size} bytes)")
            
        except Exception as e:
            logger.log_error("csv_export", e, output_path=path)
            raise


def export_markdown(results: TestResults, path: str) -> None:
    logger = get_logger()
    
    with log_operation("markdown_export", format="markdown", output_path=path):
        try:
            logger.debug(f"Exporting results to Markdown: {path}")
            
            lines = [f"# RAGTest Results", "", f"Overall score: {results.overall_score():.3f}", ""]
            total_questions = 0
            
            for cat, sc in results.scorecards.items():
                lines.append(f"## {cat.value} (avg {sc.average_score:.3f})")
                for e in sc.evaluations[:10]:
                    lines.append(f"- Q: {e.question.text}")
                    lines.append(f"  - score: {e.score:.2f}, verdict: {e.verdict}")
                    if e.reasoning:
                        lines.append(f"  - reasoning: {e.reasoning}")
                    total_questions += 1
                lines.append("")
            
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            
            file_size = os.path.getsize(path)
            logger.debug(f"Markdown export completed: {total_questions} questions written to {path} ({file_size} bytes)")
            
        except Exception as e:
            logger.log_error("markdown_export", e, output_path=path)
            raise


def export_html(results: TestResults, path: str) -> None:
    logger = get_logger()
    
    with log_operation("html_export", format="html", output_path=path):
        try:
            logger.debug(f"Exporting results to HTML: {path}")
            
            # Simple static HTML
            rows = []
            total_questions = 0
            
            for cat, sc in results.scorecards.items():
                category_html = f"<h2>{cat.value} (avg {sc.average_score:.3f})</h2>"
                questions_html = "<ul>" + "".join(
                    f"<li><b>Q:</b> {e.question.text} — <i>{e.score:.2f}</i> ({e.verdict})"
                    + (f"<br><small><b>Reasoning:</b> {e.reasoning}</small>" if e.reasoning else "")
                    + "</li>" 
                    for e in sc.evaluations[:20]
                ) + "</ul>"
                rows.append(category_html + questions_html)
                total_questions += min(len(sc.evaluations), 20)
            
            html = (
                "<html><head><meta charset='utf-8'><title>RAGTest Results</title>" 
                "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 12px}</style>" 
                "</head><body>" 
                f"<h1>Overall score: {results.overall_score():.3f}</h1>" 
                + "".join(rows) + "</body></html>"
            )
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            
            file_size = os.path.getsize(path)
            logger.debug(f"HTML export completed: {total_questions} questions written to {path} ({file_size} bytes)")
            
        except Exception as e:
            logger.log_error("html_export", e, output_path=path)
            raise


