"""RAGAS evaluation configuration and pipeline scorer."""

from __future__ import annotations

from typing import Literal

from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

JUDGE_MODEL = "claude-haiku-4-5"

MetricSet = Literal["baseline", "full"]

BASELINE_METRICS = [faithfulness, answer_relevancy]
FULL_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


class RawResult(BaseModel):
    """Output of running one test case through the graph."""

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
                for metric in ["faithfulness", "answer_relevancy",
                               "context_precision", "context_recall"]:
                    val = getattr(s, metric)
                    if val is not None:
                        scores[metric].append(val)
            result[category] = {
                k: round(sum(v) / len(v), 3)
                for k, v in scores.items()
            }
        return result


def _build_judge_llm() -> LangchainLLMWrapper:
    return LangchainLLMWrapper(
        ChatAnthropic(model=JUDGE_MODEL, temperature=0)
    )


def _to_ragas_dataset(results: list[RawResult]) -> Dataset:
    """Convert raw results to a RAGAS-compatible HuggingFace Dataset."""
    return Dataset.from_dict(
        {
            "question": [r.question for r in results],
            "answer": [r.response for r in results],
            "contexts": [r.retrieved_contexts for r in results],
            "ground_truth": [r.expected_answer for r in results],
        }
    )


def evaluate_pipeline(
    results: list[RawResult],
    metric_set: MetricSet = "baseline",
) -> EvaluationReport:
    """
    Run RAGAS evaluation on raw pipeline results.
    Includes fallback cases — they will score poorly by design.
    Returns both aggregate and per-sample scores.
    """
    metrics = BASELINE_METRICS if metric_set == "baseline" else FULL_METRICS
    judge = _build_judge_llm()

    for metric in metrics:
        metric.llm = judge

    dataset = _to_ragas_dataset(results)
    ragas_result = evaluate(dataset, metrics=metrics)

    # Extract per-sample scores
    scores_df = ragas_result.to_pandas()
    per_sample = []
    for i, result in enumerate(results):
        row = scores_df.iloc[i]
        per_sample.append(SampleScore(
            test_case_id=result.test_case_id,
            question=result.question,
            category=result.category,
            fallback_triggered=result.fallback_triggered,
            faithfulness=row.get("faithfulness"),
            answer_relevancy=row.get("answer_relevancy"),
            context_precision=row.get("context_precision"),
            context_recall=row.get("context_recall"),
        ))

    # Aggregate scores
    aggregate = {
        col: round(float(scores_df[col].mean()), 3)
        for col in scores_df.columns
        if col in {"faithfulness", "answer_relevancy",
                   "context_precision", "context_recall"}
    }

    return EvaluationReport(
        metric_set=metric_set,
        total_cases=len(results),
        fallback_count=sum(1 for r in results if r.fallback_triggered),
        aggregate=aggregate,
        per_sample=per_sample,
    )