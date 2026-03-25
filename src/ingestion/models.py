"""Pydantic models for chunk metadata; SQLAlchemy models for persistence."""

from __future__ import annotations

from typing import Literal

from pgvector.sqlalchemy import Vector
from pydantic import BaseModel
from sqlalchemy import String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DocumentType = Literal["city_guide", "visa_info", "coworking_review", "cost_comparison"]


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for ORM tables."""


class ChunkMetadata(BaseModel):
    """Structured metadata attached to each chunk for filtering and attribution."""

    source_file: str
    document_type: DocumentType
    country: str | None = None
    city: str | None = None
    section: str | None = None
    chunk_strategy: str


class ChunkRecord(Base):
    """Stored chunk row for pgvector similarity search and metadata filtering.

    ``text_search`` (``tsvector``) is not mapped: PostgreSQL trigger
    ``chunks_tsvector_update`` fills it on every INSERT and UPDATE from ``text``,
    so the ORM never sends that column. For SQL/Core filters, use
    ``ChunkRecord.__table__.c.text_search``.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)
    source_file: Mapped[str] = mapped_column(Text, nullable=False)
    document_type: Mapped[str] = mapped_column(Text, nullable=False)
    country: Mapped[str | None] = mapped_column(Text, nullable=True)
    city: Mapped[str | None] = mapped_column(Text, nullable=True)
    section: Mapped[str | None] = mapped_column(Text, nullable=True)
    chunk_strategy: Mapped[str] = mapped_column(Text, nullable=False)
