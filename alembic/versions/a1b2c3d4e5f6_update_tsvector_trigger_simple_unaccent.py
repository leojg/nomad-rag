"""update tsvector trigger to use simple config with unaccent

Revision ID: a1b2c3d4e5f6
Revises: dc4045f226bf
Create Date: 2026-03-24 14:30:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "dc4045f226bf"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Enable unaccent extension
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS unaccent"))

    # 2. Drop existing trigger
    op.execute(sa.text("DROP TRIGGER IF EXISTS chunks_tsvector_update ON chunks"))

    # 3. Recreate with simple config (handles accents and proper nouns better)
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

    # 4. Re-index all existing rows by touching the text column
    op.execute(sa.text("UPDATE chunks SET text = text"))


def downgrade() -> None:
    op.execute(sa.text("DROP TRIGGER IF EXISTS chunks_tsvector_update ON chunks"))
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
    op.execute(sa.text("UPDATE chunks SET text = text"))
