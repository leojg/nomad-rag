"""Transport-agnostic chat service wrapping the LangGraph agent."""

from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph
from models.chat import ChatResponse, SourceAttribution

from ingestion.models import ChunkRecord


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
    """Run the RAG agent and return a structured response."""
    from agent.prompts import FALLBACK_RESPONSE

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
