"""Transport-agnostic chat service wrapping the LangGraph chain."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from ingestion.models import ChunkRecord

class SourceAttribution(BaseModel):
    source_file: str
    section: str | None
    document_type: str


class ChatResponse(BaseModel):
    response: str
    sources: list[SourceAttribution]
    fallback_triggered: bool


def _extract_sources(chunks: list[ChunkRecord]) -> list[SourceAttribution]:
    seen = set()
    sources = []
    for chunk in chunks:
        key = (chunk.source_file, chunk.section)
        if key not in seen:
            seen.add(key)
            sources.append(SourceAttribution(
                source_file=chunk.source_file,
                section=chunk.section,
                document_type=chunk.document_type,
            ))
    return sources


def chat_service(
    query: str,
    filters: dict[str, str] | None,
    graph: CompiledStateGraph,
) -> ChatResponse:
    """Run the RAG chain and return a structured response."""
    from chain.prompts import FALLBACK_RESPONSE

    state = graph.invoke({
        "query": query,
        "filters": filters,
        "parsed_queries": None,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "response": "",
    })

    reranked = state.get("reranked_chunks") or []
    response = state.get("response") or ""
    fallback_triggered = not reranked or response == FALLBACK_RESPONSE

    return ChatResponse(
        response=response,
        sources=_extract_sources(reranked),
        fallback_triggered=fallback_triggered,
    )