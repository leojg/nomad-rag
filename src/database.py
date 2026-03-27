"""SQLAlchemy engine and sessions for PostgreSQL + pgvector.

Callers typically create one :func:`create_db_engine` at startup, pass ``engine`` into
services, and use :func:`create_session_factory` or :func:`session_scope` where needed.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from config.settings import DEFAULT_DATABASE_URL
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

_REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_REPO_ROOT / ".env")


def get_database_url() -> str:
    """Return ``DATABASE_URL`` from the environment, or :data:`DEFAULT_DATABASE_URL`."""
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def create_db_engine(url: str | None = None, **kwargs: Any) -> Engine:
    """Create a SQLAlchemy :class:`~sqlalchemy.engine.Engine`.

    ``kwargs`` are forwarded to :func:`sqlalchemy.create_engine` (e.g. ``pool_size``).
    ``pool_pre_ping`` defaults to ``True`` unless overridden.
    """
    kwargs.setdefault("pool_pre_ping", True)
    return create_engine(url or get_database_url(), **kwargs)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine`` for use in consuming code."""
    return sessionmaker(bind=engine, autocommit=False, autoflush=False, class_=Session)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Context manager: one Session, commit on success, rollback on error, always close."""
    factory = create_session_factory(engine)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = [
    "DEFAULT_DATABASE_URL",
    "create_db_engine",
    "create_session_factory",
    "get_database_url",
    "session_scope",
]
