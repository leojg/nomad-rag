"""Configuration for the LangGraph RAG chain."""

from __future__ import annotations

from ingestion.vector_store import EMBEDDING_MODEL
from dataclasses import dataclass

@dataclass
class GraphConfig:
    """Graph configuration."""

    model: str = "claude-sonnet-4-6"
    embeddings_model: str = EMBEDDING_MODEL
    temperature: float = 0.0

    retrieve_k: int = 10 # number of chunks to retrieve
    rerank_k: int = 4 # number of chunks to rerank
