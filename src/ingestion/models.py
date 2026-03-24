"""Pydantic models for chunk metadata and ingestion."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

DocumentType = Literal["city_guide", "visa_info", "coworking_review", "cost_comparison"]


class ChunkMetadata(BaseModel):
    """Structured metadata attached to each chunk for filtering and attribution."""

    source_file: str
    document_type: DocumentType
    country: str | None = None
    city: str | None = None
    section: str | None = None
    chunk_strategy: str
