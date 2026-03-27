"""Pydantic models for evaluation datasets and RAGAS scoring."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

QueryCategory = Literal["factual", "comparison", "aggregation", "out_of_scope", "ambiguous"]

MetricSet = Literal["baseline", "full"]


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


class RawResult(BaseModel):
    """Output of running one test case through the agent graph."""

    test_case_id: str
    question: str
    expected_answer: str
    category: str
    response: str
    retrieved_contexts: list[str]
    fallback_triggered: bool


class SampleScore(BaseModel):
    """RAGAS scores for a single test case."""

    test_case_id: str
    question: str
    category: str
    fallback_triggered: bool
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None


class EvaluationReport(BaseModel):
    """Aggregate and per-sample scores from a full evaluation run."""

    metric_set: MetricSet
    total_cases: int
    fallback_count: int
    aggregate: dict[str, float]
    per_sample: list[SampleScore]

    @property
    def by_category(self) -> dict[str, dict[str, float]]:
        """Aggregate scores grouped by query category."""
        from collections import defaultdict

        buckets: dict[str, list[SampleScore]] = defaultdict(list)
        for s in self.per_sample:
            buckets[s.category].append(s)

        result = {}
        for category, samples in buckets.items():
            scores: dict[str, list[float]] = defaultdict(list)
            for s in samples:
                for metric in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]:
                    val = getattr(s, metric)
                    if val is not None:
                        scores[metric].append(val)
            result[category] = {
                k: round(sum(v) / len(v), 3)
                for k, v in scores.items()
            }
        return result
