#!/usr/bin/env python3
"""Smoke-test the LangGraph RAG chain against the live database.

Run from the repo root:
  venv/bin/python scripts/test_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv

from chain.config import GraphConfig
from chain.graph import make_graph
from database import create_db_engine, session_scope

load_dotenv(ROOT / ".env")

# Same scenarios as scripts/test_retrieval.py (k aligned via GraphConfig below).
TEST_QUERIES = [
    {
        "query": "coworking spaces with fast internet",
        "filters": {"document_type": "coworking_review"},
        "label": "Coworking — filtered by document_type",
    },
    {
        "query": "digital nomad visa requirements Colombia",
        "filters": {"country": "Colombia"},
        "label": "Visa — filtered by country",
    },
    {
        "query": "cost of rent",
        "filters": None,
        "label": "Cost — no filter",
    },
    {
        "query": "FMM tourist permit 180 days Mexico",
        "filters": {"document_type": "visa_info"},
        "label": "Keyword-heavy — exact terms",
    },
    {
        "query": "best ski resorts in the Alps",
        "filters": None,
        "label": "Out of scope — should trigger fallback",
    },
    {
        "query": "compare cost of living between Medellín and Buenos Aires",
        "filters": None,
        "label": "Multi-entity — should trigger multi_retrieve",
    },
    {
        "query": "coworking spaces in Mexico City vs Buenos Aires",
        "filters": None,
        "label": "Multi-entity comparison — coworking",
    },
]


def _initial_state(query: str, filters: dict[str, str] | None) -> dict:
    """Full initial state required by chain.state.State."""
    return {
        "query": query,
        "filters": filters,
        "parsed_queries": None,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "response": "",
    }


def print_run(label: str, result: dict) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"{'─' * 60}")
    parsed = result.get("parsed_queries") or []
    print(f"  parsed intents: {len(parsed)}")
    for i, intent in enumerate(parsed, start=1):
        print(f"    [{i}] query='{intent['query']}' filters={intent['filters']}")
    retrieved = result.get("retrieved_chunks") or []
    reranked = result.get("reranked_chunks") or []
    print(f"  retrieved: {len(retrieved)} → reranked: {len(reranked)}")
    print()
    print(result.get("response"))
    
def main() -> None:
    engine = create_db_engine()
    # Match retrieval smoke test k=3 for comparable breadth.
    config = GraphConfig(retrieve_k=6, rerank_k=3)

    with session_scope(engine) as session:
        graph = make_graph(config, session)
        for test in TEST_QUERIES:
            result = graph.invoke(_initial_state(test["query"], test["filters"]))
            print_run(test["label"], result)


if __name__ == "__main__":
    main()
