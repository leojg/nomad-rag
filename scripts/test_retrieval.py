#!/usr/bin/env python3
"""Smoke-test semantic and keyword search against the live database.

Run from the repo root:
  venv/bin/python scripts/test_retrieval.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from database import create_db_engine, session_scope
from ingestion.vector_store import EMBEDDING_MODEL, similarity_search
from retrieval.keyword_search import keyword_search
from retrieval.hybrid import hybrid_search

load_dotenv(ROOT / ".env")

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
]

K = 3


def print_results(label: str, strategy: str, results: list) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {label}")
    print(f"  Strategy: {strategy} | k={K}")
    print(f"{'─' * 60}")
    for i, r in enumerate(results, start=1):
        print(f"  [{i}] {r.source_file}")
        print(f"       section : {r.section or 'N/A'}")
        print(f"       country : {r.country or 'N/A'} | city: {r.city or 'N/A'}")
        print(f"       preview : {r.text[:120].strip().replace(chr(10), ' ')}...")


def main() -> None:
    engine = create_db_engine()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    with session_scope(engine) as session:
        for test in TEST_QUERIES:
            query = test["query"]
            filters = test["filters"]
            label = test["label"]

            # Semantic
            semantic = similarity_search(
                query=query, k=K, embeddings=embeddings,
                session=session, filters=filters,
            )
            print_results(label, "semantic", semantic)

            # Keyword
            keyword = keyword_search(
                query=query, k=K,
                session=session, filters=filters,
            )
            print_results(label, "keyword", keyword)

            # Hybrid
            hybrid = hybrid_search(
                query=query, k=K, embeddings=embeddings,
                session=session, filters=filters,
            )
            print_results(label, "hybrid", hybrid)


if __name__ == "__main__":
    main()