"""RAGAS evaluation configuration and pipeline scorer."""

from __future__ import annotations

from config.settings import RAGAS_JUDGE_MODEL
from datasets import Dataset
from langchain_anthropic import ChatAnthropic
from models.evaluation import EvaluationReport, MetricSet, RawResult, SampleScore
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

BASELINE_METRICS = [faithfulness, answer_relevancy]
FULL_METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


def _build_judge_llm() -> LangchainLLMWrapper:
    return LangchainLLMWrapper(
        ChatAnthropic(model=RAGAS_JUDGE_MODEL, temperature=0)
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
