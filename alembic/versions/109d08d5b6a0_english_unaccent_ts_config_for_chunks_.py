"""english_unaccent text search config for chunks trigger

Revision ID: 109d08d5b6a0
Revises: a1b2c3d4e5f6
Create Date: 2026-03-24 20:33:56.979656

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "109d08d5b6a0"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            CREATE TEXT SEARCH CONFIGURATION english_unaccent (COPY = pg_catalog.english)
            """
        )
    )
    op.execute(
        sa.text(
            """
            ALTER TEXT SEARCH CONFIGURATION english_unaccent
                ALTER MAPPING FOR hword, hword_part, word
                WITH unaccent, english_stem
            """
        )
    )
    op.execute(sa.text("DROP TRIGGER IF EXISTS chunks_tsvector_update ON chunks"))
    op.execute(
        sa.text(
            """
            CREATE TRIGGER chunks_tsvector_update
            BEFORE INSERT OR UPDATE ON chunks
            FOR EACH ROW EXECUTE FUNCTION
            tsvector_update_trigger('text_search', 'public.english_unaccent', 'text')
            """
        )
    )
    op.execute(sa.text("UPDATE chunks SET text = text"))


def downgrade() -> None:
    op.execute(sa.text("DROP TRIGGER IF EXISTS chunks_tsvector_update ON chunks"))
    op.execute(
        sa.text(
            """
            CREATE TRIGGER chunks_tsvector_update
            BEFORE INSERT OR UPDATE ON chunks
            FOR EACH ROW EXECUTE FUNCTION
            tsvector_update_trigger('text_search', 'pg_catalog.simple', 'text')
            """
        )
    )
    op.execute(sa.text("UPDATE chunks SET text = text"))
    op.execute(sa.text("DROP TEXT SEARCH CONFIGURATION IF EXISTS english_unaccent"))
