import hashlib

from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from config.settings import (
    OPENAI_EMBEDDING_BATCH_SIZE,
    OPENAI_EMBEDDING_DIMENSIONS,
    OPENAI_EMBEDDING_MODEL,
)
from ingestion.models import ChunkMetadata, ChunkRecord

EMBEDDING_MODEL = OPENAI_EMBEDDING_MODEL
EMBEDDING_DIMENSIONS = OPENAI_EMBEDDING_DIMENSIONS
BATCH_SIZE = OPENAI_EMBEDDING_BATCH_SIZE

def _chunk_id(source_file: str, text: str) -> str:
    """Deterministic ID from source file + content — stable across re-ingestion."""
    return hashlib.sha256(f"{source_file}:{text}".encode()).hexdigest()


def _embed_texts(texts: list[str], embeddings: OpenAIEmbeddings) -> list[list[float]]:
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        results.extend(embeddings.embed_documents(batch))
    return results

def upsert_chunks(chunks: list[tuple[str, ChunkMetadata]], embeddings: OpenAIEmbeddings, session: Session) -> int:
    """
    Embed and upsert chunks into the vector store.
    Existing rows with the same ID are updated in place.
    Returns the number of rows upserted.
    """ 

    if not chunks:
        return 0

    texts = [text for text, _ in chunks]
    embeddings = _embed_texts(texts, embeddings)

    for (text, metadata), embedding in zip(chunks, embeddings):
        if len(embedding) != OPENAI_EMBEDDING_DIMENSIONS:
            raise ValueError(
                f"Embedding length {len(embedding)} != OPENAI_EMBEDDING_DIMENSIONS "
                f"({OPENAI_EMBEDDING_DIMENSIONS}); check OPENAI_EMBEDDING_MODEL / dimensions."
            )
        chunk_id = _chunk_id(metadata.source_file, text)
        record = session.get(ChunkRecord, chunk_id)

        if record is None:
            record = ChunkRecord(id=chunk_id)
            session.add(record)
        
        record.text = text
        record.embedding = embedding
        record.source_file = metadata.source_file
        record.document_type = metadata.document_type
        record.country = metadata.country
        record.city = metadata.city
        record.section = metadata.section
        record.chunk_strategy = metadata.chunk_strategy

    return len(chunks)

def similarity_search(
    query, 
    k, 
    embeddings: OpenAIEmbeddings, 
    session: Session,
    filters: dict[str, str] | None = None
) -> list[ChunkRecord]:
    """
    Search the vector store for similar chunks.
    Returns the top k chunks.
    """

    q = session.query(ChunkRecord)
    query_embedding = embeddings.embed_query(query)

    if filters:
        for field, value in filters.items():
            q = q.filter(getattr(ChunkRecord, field) == value)


    return (
        q.order_by(ChunkRecord.embedding.cosine_distance(query_embedding)).limit(k).all()
    )

def delete_by_source(source_file: str, session: Session) -> int:
    """
    Delete chunks by source file.
    Returns the number of rows deleted.
    """

    return (
        session.query(ChunkRecord)
        .filter(ChunkRecord.source_file == source_file)
        .delete(synchronize_session=False)
    )