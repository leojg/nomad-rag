#!/usr/bin/env python3
"""Generate the evaluation test set from the actual document content.

Run once from the repo root, commit the output:
  venv/bin/python scripts/generate_test_set.py

Output: data/evaluation/test_set.json
Idempotent: re-running overwrites the same file.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anthropic import Anthropic
from dotenv import load_dotenv

from ingestion.loader import loader

load_dotenv(ROOT / ".env")

OUTPUT_PATH = ROOT / "data" / "evaluation" / "test_set.json"
GENERATION_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an evaluation dataset generator for a RAG pipeline about digital nomads in Latin America.

Given a set of source documents, generate test questions that cover these five categories:

1. factual — specific facts with a clear, verifiable answer from the documents
   Example: "What is the maximum tourist stay allowed in Colombia?"

2. comparison — questions comparing two or more cities or countries
   Example: "Is rent cheaper in Medellín or Mexico City?"

3. aggregation — questions requiring synthesis across multiple documents
   Example: "Which cities have coworking spaces under $100/month?"

4. out_of_scope — questions clearly outside the knowledge base
   Example: "What are the best coworking spaces in Tokyo?"

5. ambiguous — vague questions that require interpretation
   Example: "Where should I go as a digital nomad?"

Rules:
- factual, comparison, aggregation questions must be answerable from the provided documents
- expected_answer must be grounded in the document content — no outside knowledge
- out_of_scope expected_answer should always be a clear statement that the information is not available
- ambiguous expected_answer should acknowledge the vagueness and ask for clarification or list options
- filters should only include valid values: country (Colombia, Brazil, Mexico, Argentina), city (Medellín, Florianópolis, Mexico City, Buenos Aires), document_type (city_guide, visa_info, coworking_review, cost_comparison)

Return ONLY a raw JSON array. No markdown, no explanation. Each object must have:
- question: string
- expected_answer: string  
- category: one of factual|comparison|aggregation|out_of_scope|ambiguous
- filters: object or null
- notes: string explaining which document(s) the answer comes from (null for out_of_scope/ambiguous)"""


USER_TEMPLATE = """Generate exactly {n_total} test questions from these documents:
- {n_factual} factual
- {n_comparison} comparison
- {n_aggregation} aggregation
- {n_out_of_scope} out_of_scope
- {n_ambiguous} ambiguous

Source documents:
{documents}"""


def _format_documents(docs: list[tuple[str, object]]) -> str:
    """Format loaded documents for the generation prompt."""
    lines = []
    for text, metadata in docs:
        lines.append(f"=== {metadata.source_file} ({metadata.document_type}) ===")
        lines.append(text[:2000])  # truncate long docs to fit context
        lines.append("")
    return "\n".join(lines)


def generate_test_cases(client: Anthropic, documents: str) -> list[dict]:
    """Call Claude to generate test cases from the documents."""
    response = client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": USER_TEMPLATE.format(
                    n_total=25,
                    n_factual=6,
                    n_comparison=6,
                    n_aggregation=5,
                    n_out_of_scope=4,
                    n_ambiguous=4,
                    documents=documents,
                ),
            }
        ],
    )
    content = response.content[0].text.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    return json.loads(content)


def build_test_set(cases: list[dict]) -> dict:
    """Wrap raw cases in the TestSet envelope."""
    validated = []
    for i, case in enumerate(cases):
        validated.append(
            {
                "id": f"tc_{str(uuid.uuid4())[:8]}",
                "question": case["question"],
                "expected_answer": case["expected_answer"],
                "category": case["category"],
                "filters": case.get("filters") or None,
                "notes": case.get("notes"),
            }
        )
    return {
        "version": "1.0",
        "generated_by": GENERATION_MODEL,
        "cases": validated,
    }


def main() -> None:
    client = Anthropic()

    print("Loading documents...")
    docs = loader()
    print(f"  {len(docs)} documents loaded.")

    print("Formatting documents for prompt...")
    formatted = _format_documents(docs)

    print(f"Generating test cases with {GENERATION_MODEL}...")
    raw_cases = generate_test_cases(client, formatted)
    print(f"  {len(raw_cases)} cases generated.")

    test_set = build_test_set(raw_cases)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(test_set, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Test set written to {OUTPUT_PATH}")

    # Print category breakdown
    by_category: dict[str, int] = {}
    for case in test_set["cases"]:
        by_category[case["category"]] = by_category.get(case["category"], 0) + 1
    print("\nCategory breakdown:")
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count}")


if __name__ == "__main__":
    main()