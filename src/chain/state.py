from typing import TypedDict
from ingestion.models import ChunkRecord

class State(TypedDict):
    query: str
    retrieved_chunks: list[ChunkRecord] 
    reranked_chunks: list[ChunkRecord]
    filters: dict[str, str] | None  # passed to retrieve, not used downstream
    response: str | None = None
