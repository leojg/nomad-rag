"""API models for document ingestion."""

from __future__ import annotations

from pydantic import BaseModel


class IngestResponse(BaseModel):
    source_file: str
    chunks_upserted: int
    document_type: str
