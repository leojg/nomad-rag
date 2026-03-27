"""Central defaults and environment-backed settings (non-secret defaults only)."""

from __future__ import annotations

import os


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return int(raw)


# Anthropic (RAG agent)
ANTHROPIC_GENERATE_MODEL = _env("ANTHROPIC_GENERATE_MODEL", "claude-sonnet-4-6")
ANTHROPIC_RERANK_MODEL = _env("ANTHROPIC_RERANK_MODEL", "claude-haiku-4-5")
ANTHROPIC_ANALYSIS_MODEL = _env("ANTHROPIC_ANALYSIS_MODEL", "claude-haiku-4-5")

# OpenAI embeddings
OPENAI_EMBEDDING_MODEL = _env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIMENSIONS = _env_int("OPENAI_EMBEDDING_DIMENSIONS", 1536)
OPENAI_EMBEDDING_BATCH_SIZE = _env_int("OPENAI_EMBEDDING_BATCH_SIZE", 100)

# Retrieval
RETRIEVAL_RRF_K = _env_int("RETRIEVAL_RRF_K", 60)

# RAGAS judge LLM (evaluation)
RAGAS_JUDGE_MODEL = _env("RAGAS_JUDGE_MODEL", "claude-haiku-4-5")

# PostgreSQL (used when DATABASE_URL is unset)
DEFAULT_DATABASE_URL = _env(
    "DEFAULT_DATABASE_URL",
    "postgresql+psycopg2://nomad:nomad@localhost:5433/nomad_latam",
)
