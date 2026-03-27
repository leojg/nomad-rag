"""Configuration for the LangGraph RAG agent."""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import (
    ANTHROPIC_ANALYSIS_MODEL,
    ANTHROPIC_GENERATE_MODEL,
    ANTHROPIC_RERANK_MODEL,
    OPENAI_EMBEDDING_MODEL,
)


@dataclass
class AgentConfig:
    """LLM and retrieval settings for the agent graph."""

    generate_model: str = ANTHROPIC_GENERATE_MODEL
    rerank_model: str = ANTHROPIC_RERANK_MODEL
    analysis_model: str = ANTHROPIC_ANALYSIS_MODEL
    embeddings_model: str = OPENAI_EMBEDDING_MODEL
    temperature: float = 0.0

    retrieve_k: int = 6
    rerank_k: int = 3
