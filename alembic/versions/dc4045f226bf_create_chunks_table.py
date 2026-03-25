"""create chunks table indexes and tsvector trigger

Revision ID: dc4045f226bf
Revises:
Create Date: 2026-03-24 13:59:50.085289

"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "dc4045f226bf"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS vector"))

    # 1. Create the chunks table with all columns including text_search
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("source_file", sa.Text(), nullable=False),
        sa.Column("document_type", sa.Text(), nullable=False),
        sa.Column("country", sa.Text(), nullable=True),
        sa.Column("city", sa.Text(), nullable=True),
        sa.Column("section", sa.Text(), nullable=True),
        sa.Column("chunk_strategy", sa.Text(), nullable=False),
        sa.Column("text_search", postgresql.TSVECTOR(), nullable=True),
    )

    # 2. Indexes
    op.create_index("chunks_source_file_idx", "chunks", ["source_file"])
    op.create_index("chunks_document_type_idx", "chunks", ["document_type"])
    op.create_index("chunks_country_idx", "chunks", ["country"])
    op.create_index("chunks_city_idx", "chunks", ["city"])
    op.execute(
        sa.text(
            "CREATE INDEX chunks_embedding_idx ON chunks "
            "USING hnsw (embedding vector_cosine_ops)"
        )
    )
    op.execute(
        sa.text(
            "CREATE INDEX chunks_text_search_idx ON chunks USING GIN (text_search)"
        )
    )

    # 3. Trigger — fills text_search from text on every INSERT and UPDATE (ORM omits text_search)
    op.execute(
        sa.text(
            """
            CREATE TRIGGER chunks_tsvector_update
            BEFORE INSERT OR UPDATE ON chunks
            FOR EACH ROW EXECUTE FUNCTION
            tsvector_update_trigger('text_search', 'pg_catalog.english', 'text')
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DROP TRIGGER IF EXISTS chunks_tsvector_update ON chunks"))
    op.drop_table("chunks")
