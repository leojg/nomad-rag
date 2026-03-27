"""Test case definitions and loader for the evaluation framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent.parent
TEST_SET_PATH = ROOT / "data" / "evaluation" / "test_set.json"

QueryCategory = Literal["factual", "comparison", "aggregation", "out_of_scope", "ambiguous"]


class TestCase(BaseModel):
    """A single evaluation test case."""

    id: str
    question: str
    expected_answer: str
    category: QueryCategory
    filters: dict[str, str] | None = None
    notes: str | None = None


class TestSet(BaseModel):
    """Collection of test cases with metadata."""

    version: str
    generated_by: str
    cases: list[TestCase]

    @property
    def by_category(self) -> dict[QueryCategory, list[TestCase]]:
        result: dict[str, list[TestCase]] = {}
        for case in self.cases:
            result.setdefault(case.category, []).append(case)
        return result

    def sample(self, n: int) -> list[TestCase]:
        """Return up to n cases, distributed across categories."""
        categories = self.by_category
        per_category = max(1, n // len(categories))
        sampled = []
        for cases in categories.values():
            sampled.extend(cases[:per_category])
        return sampled[:n]


def load_test_set(path: Path = TEST_SET_PATH) -> TestSet:
    """Load and validate the test set from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Test set not found at {path}. "
            "Run scripts/generate_test_set.py first."
        )
    raw = json.loads(path.read_text(encoding="utf-8"))
    return TestSet.model_validate(raw)