"""API and transport models for chat."""

from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    filters: dict[str, str] | None = None


class SourceAttribution(BaseModel):
    source_file: str
    section: str | None
    document_type: str


class ChatResponse(BaseModel):
    response: str
    sources: list[SourceAttribution]  # extracted from reranked_chunks
    fallback_triggered: bool
