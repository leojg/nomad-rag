"""Hybrid retrieval combining semantic similarity and PostgreSQL full-text (tsvector) search via RRF."""

from __future__ import annotations

from collections import defaultdict

from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from config.settings import RETRIEVAL_RRF_K
from ingestion.models import ChunkRecord
from ingestion.vector_store import similarity_search
from retrieval.keyword_search import keyword_search

RRF_K = RETRIEVAL_RRF_K


def reciprocal_rank_fusion(
    ranked_lists: list[list[ChunkRecord]],
) -> list[ChunkRecord]:
    """Merge ranked lists via Reciprocal Rank Fusion, return deduplicated results."""
    scores: dict[str, float] = defaultdict(float)
    records: dict[str, ChunkRecord] = {}

    for ranked in ranked_lists:
        for rank, record in enumerate(ranked, start=1):
            scores[record.id] += 1.0 / (rank + RRF_K)
            records[record.id] = record

    return sorted(records.values(), key=lambda r: scores[r.id], reverse=True)


def hybrid_search(
    query: str,
    k: int,
    embeddings: OpenAIEmbeddings,
    session: Session,
    filters: dict[str, str] | None = None,
) -> list[ChunkRecord]:
    """
    Run semantic and keyword search in parallel, merge via RRF, return top-k chunks.
    Each sub-search fetches k*2 candidates before merging.
    """
    candidate_k = k * 2

    semantic_results = similarity_search(
        query=query,
        k=candidate_k,
        embeddings=embeddings,
        session=session,
        filters=filters,
    )
    keyword_results = keyword_search(
        query=query,
        k=candidate_k,
        session=session,
        filters=filters,
    )

    merged = reciprocal_rank_fusion([semantic_results, keyword_results])
    return merged[:k]