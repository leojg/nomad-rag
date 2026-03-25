from sqlalchemy import func, literal_column
from sqlalchemy.orm import Session
from ingestion.models import ChunkRecord

def keyword_search(
    query: str,
    k: int,
    session: Session,
    filters: dict[str, str] | None = None,
) -> list[ChunkRecord]:
    """
    Full-text keyword search using PostgreSQL tsvector/tsquery.
    Returns the top-k chunks ranked by ts_rank.
    """

    ts_query = func.plainto_tsquery(literal_column("'english_unaccent'"), query)
    text_search_col = literal_column("text_search")
    
    q = session.query(ChunkRecord).filter(
        text_search_col.op("@@")(ts_query)
    )

    if filters:
        for field, value in filters.items():
            q = q.filter(getattr(ChunkRecord, field) == value)

    return (
        q.order_by(func.ts_rank(text_search_col, ts_query).desc())
        .limit(k)
        .all()
    )