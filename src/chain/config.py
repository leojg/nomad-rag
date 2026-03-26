"""Configuration for the LangGraph RAG chain."""

from __future__ import annotations

from ingestion.vector_store import EMBEDDING_MODEL
from dataclasses import dataclass

@dataclass
class GraphConfig:
    """Graph configuration."""

    model: str = "claude-sonnet-4-6"
    rerank_model: str = "claude-haiku-4-5"
    embeddings_model: str = EMBEDDING_MODEL
    temperature: float = 0.0

    retrieve_k: int = 6 # number of chunks to retrieve
    rerank_k: int = 3 # number of chunks to rerank
