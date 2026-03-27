#!/usr/bin/env python3
"""Ingest all markdown documents into the vector store.

Run from the repo root:
  venv/bin/python scripts/ingest.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from database import create_db_engine, session_scope
from ingestion.chunking import MARKDOWN_HEADERS_TO_SPLIT_ON, MarkdownHeaderTextSplitterStrategy
from ingestion.loader import loader
from ingestion.vector_store import EMBEDDING_MODEL, upsert_chunks

load_dotenv(ROOT / ".env")


def _check_env() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set.", file=sys.stderr)
        sys.exit(1)


def _build_strategy() -> MarkdownHeaderTextSplitterStrategy:
    return MarkdownHeaderTextSplitterStrategy(
        headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
    )


def main() -> None:
    _check_env()

    engine = create_db_engine()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    strategy = _build_strategy()

    print("Loading documents...")
    documents = loader()
    print(f"  {len(documents)} documents found.")

    print("Chunking...")
    all_chunks = []
    for text, metadata in documents:
        chunks = strategy.chunk(text, metadata)
        chunks = [(t, m) for t, m in chunks if t.strip()]
        all_chunks.extend(chunks)
    print(f"  {len(all_chunks)} chunks produced.")

    print("Upserting into vector store...")
    with session_scope(engine) as session:
        n = upsert_chunks(all_chunks, embeddings, session)
    print(f"  {n} chunks upserted.")

    print("Done.")


if __name__ == "__main__":
    main()