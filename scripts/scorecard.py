#!/usr/bin/env python3
"""Run the evaluation test set through the RAG pipeline and score with RAGAS.

Usage:
  # Full run
  venv/bin/python scripts/scorecard.py

  # Quick check — 5 samples, baseline metrics only
  venv/bin/python scripts/scorecard.py --samples 5 --metrics baseline

  # Skip graph inference, re-score existing raw results
  venv/bin/python scripts/scorecard.py --skip-inference --metrics full
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from chain.config import GraphConfig
from chain.graph import make_graph
from database import create_db_engine, session_scope
from evaluation.metrics import EvaluationReport, RawResult, evaluate_pipeline
from evaluation.test_set import TestCase, load_test_set

load_dotenv(ROOT / ".env")

RAW_RESULTS_PATH = ROOT / "data" / "evaluation" / "raw_results.json"
RESULTS_DIR = ROOT / "data" / "evaluation" / "results"


# ---------------------------------------------------------------------------
# Graph inference
# ---------------------------------------------------------------------------

def _initial_state(case: TestCase) -> dict:
    return {
        "query": case.question,
        "filters": case.filters,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "parsed_queries": [],
        "response": None,
    }


def run_inference(cases: list[TestCase]) -> list[RawResult]:
    """Run each test case through the compiled graph and collect raw results."""
    engine = create_db_engine()
    config = GraphConfig()
    results = []

    with session_scope(engine) as session:
        graph = make_graph(config, session)

        for i, case in enumerate(cases, start=1):
            print(f"  [{i}/{len(cases)}] {case.category} — {case.question[:60]}...")
            state = graph.invoke(_initial_state(case))

            reranked = state.get("reranked_chunks") or []
            retrieved_contexts = [chunk.text for chunk in reranked]
            response = state.get("response") or ""
            fallback_triggered = (
                not state.get("reranked_chunks")
                or response == _fallback_response()
            )

            results.append(RawResult(
                test_case_id=case.id,
                question=case.question,
                expected_answer=case.expected_answer,
                category=case.category,
                response=response,
                retrieved_contexts=retrieved_contexts,
                fallback_triggered=fallback_triggered,
            ))

    return results


def _fallback_response() -> str:
    from chain.prompts import FALLBACK_RESPONSE
    return FALLBACK_RESPONSE


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_raw_results(results: list[RawResult]) -> None:
    RAW_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAW_RESULTS_PATH.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Raw results saved to {RAW_RESULTS_PATH}")


def load_raw_results() -> list[RawResult]:
    if not RAW_RESULTS_PATH.exists():
        print("No raw results found. Run without --skip-inference first.", file=sys.stderr)
        sys.exit(1)
    raw = json.loads(RAW_RESULTS_PATH.read_text(encoding="utf-8"))
    return [RawResult.model_validate(r) for r in raw]


def save_report(report: EvaluationReport, tag: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"result_{timestamp}_{tag}.json"
    path.write_text(
        json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(report: EvaluationReport) -> None:
    print(f"\n{'═' * 60}")
    print(f"  Evaluation Report — {report.metric_set} metrics")
    print(f"{'═' * 60}")
    print(f"  Total cases : {report.total_cases}")
    print(f"  Fallbacks   : {report.fallback_count}")
    print()

    print("  Aggregate scores:")
    for metric, score in report.aggregate.items():
        print(f"    {metric:<25} {score:.3f}")

    print()
    print("  By category:")
    for category, scores in report.by_category.items():
        print(f"    {category}")
        for metric, score in scores.items():
            print(f"      {metric:<23} {score:.3f}")

    print()
    print("  Per-sample (sorted by faithfulness):")
    sorted_samples = sorted(
        report.per_sample,
        key=lambda s: s.faithfulness or 0.0,
    )
    for s in sorted_samples:
        fallback_flag = " [FALLBACK]" if s.fallback_triggered else ""
        faith = f"{s.faithfulness:.2f}" if s.faithfulness is not None else "N/A"
        rel = f"{s.answer_relevancy:.2f}" if s.answer_relevancy is not None else "N/A"
        print(f"    [{s.category}]{fallback_flag}")
        print(f"      Q: {s.question[:70]}")
        print(f"      faithfulness={faith}  answer_relevancy={rel}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation.")
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of test cases to run (default: all)",
    )
    parser.add_argument(
        "--metrics",
        choices=["baseline", "full"],
        default="baseline",
        help="Metric set to run (default: baseline)",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip graph inference and reuse existing raw_results.json",
    )
    parser.add_argument(
        "--tag",
        default="run",
        help="Tag for the result snapshot filename (e.g. m3, m4)",
    )
    args = parser.parse_args()

    # Phase 1 — graph inference
    if args.skip_inference:
        print("Loading existing raw results...")
        results = load_raw_results()
        if args.samples:
            results = results[: args.samples]
    else:
        print("Loading test set...")
        test_set = load_test_set()
        cases = (
            test_set.sample(args.samples)
            if args.samples
            else test_set.cases
        )
        print(f"  {len(cases)} cases selected.")

        print("\nPhase 1 — running graph inference...")
        results = run_inference(cases)
        save_raw_results(results)

    # Phase 2 — RAGAS scoring
    print(f"\nPhase 2 — running RAGAS evaluation ({args.metrics} metrics)...")
    report = evaluate_pipeline(results, metric_set=args.metrics)

    # Save and print
    snapshot_path = save_report(report, tag=args.tag)
    print(f"  Report saved to {snapshot_path}")

    print_report(report)


if __name__ == "__main__":
    main()