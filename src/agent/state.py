from typing import TypedDict

from ingestion.models import ChunkRecord


class QueryIntent(TypedDict):
    """Rewritten or decomposed query plus optional filters for that sub-query."""

    query: str
    filters: dict[str, str] | None


class State(TypedDict):
    query: str  # original, never mutated
    filters: dict[str, str] | None
    parsed_queries: list[QueryIntent] | None  # set by query_analysis
    retrieved_chunks: list[ChunkRecord]
    reranked_chunks: list[ChunkRecord]
    response: str
