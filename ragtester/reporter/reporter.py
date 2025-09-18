from __future__ import annotations

import csv
import json
from dataclasses import asdict
from typing import Dict

from ..types import CategoryScorecard, TestCategory, TestResults


def to_json(results: TestResults) -> str:
    data = {
        "overall_score": results.overall_score(),
        "scorecards": {
            cat.value: {
                "average_score": sc.average_score,
                "evaluations": [
                    {
                        "question": e.question.text,
                        "category": e.question.category.value,
                        "score": e.score,
                        "verdict": e.verdict,
                        "reasoning": e.reasoning,
                    }
                    for e in sc.evaluations
                ],
            }
            for cat, sc in results.scorecards.items()
        },
    }
    return json.dumps(data, indent=2, ensure_ascii=False)


def to_console(results: TestResults) -> str:
    lines = []
    lines.append(f"Overall score: {results.overall_score():.3f}")
    for cat, sc in results.scorecards.items():
        lines.append(f"- {cat.value}: avg={sc.average_score:.3f} ({len(sc.evaluations)} items)")
    return "\n".join(lines)


def export_csv(results: TestResults, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "question", "score", "verdict", "reasoning"])
        for cat, sc in results.scorecards.items():
            for e in sc.evaluations:
                writer.writerow([cat.value, e.question.text, e.score, e.verdict, e.reasoning])


def export_markdown(results: TestResults, path: str) -> None:
    lines = [f"# RAGTest Results", "", f"Overall score: {results.overall_score():.3f}", ""]
    for cat, sc in results.scorecards.items():
        lines.append(f"## {cat.value} (avg {sc.average_score:.3f})")
        for e in sc.evaluations[:10]:
            lines.append(f"- Q: {e.question.text}")
            lines.append(f"  - score: {e.score:.2f}, verdict: {e.verdict}")
        lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def export_html(results: TestResults, path: str) -> None:
    # Simple static HTML
    rows = []
    for cat, sc in results.scorecards.items():
        rows.append(
            f"<h2>{cat.value} (avg {sc.average_score:.3f})</h2>" +
            "<ul>" + "".join(
                f"<li><b>Q:</b> {e.question.text} — <i>{e.score:.2f}</i> ({e.verdict})</li>" for e in sc.evaluations[:20]
            ) + "</ul>"
        )
    html = (
        "<html><head><meta charset='utf-8'><title>RAGTest Results</title>" 
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 12px}</style>" 
        "</head><body>" 
        f"<h1>Overall score: {results.overall_score():.3f}</h1>" 
        + "".join(rows) + "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


