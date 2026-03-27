"""Transport-agnostic document ingestion service."""

from __future__ import annotations

from pathlib import Path

from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ingestion.chunking import MarkdownHeaderTextSplitterStrategy
from ingestion.loader import load_file
from ingestion.vector_store import upsert_chunks

HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


class IngestResponse(BaseModel):
    source_file: str
    chunks_upserted: int
    document_type: str


def ingest_document(
    path: Path,
    embeddings: OpenAIEmbeddings,
    session: Session,
) -> IngestResponse:
    """Load, chunk, embed and upsert a single markdown document."""
    text, metadata = load_file(path)

    strategy = MarkdownHeaderTextSplitterStrategy(
        headers_to_split_on=HEADERS_TO_SPLIT_ON
    )
    chunks = strategy.chunk(text, metadata)
    chunks = [(t, m) for t, m in chunks if t.strip()]

    n = upsert_chunks(chunks, embeddings, session)

    return IngestResponse(
        source_file=metadata.source_file,
        chunks_upserted=n,
        document_type=metadata.document_type,
    )