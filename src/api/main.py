"""FastAPI application entry point with lifespan state management."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from chain.config import GraphConfig
from chain.graph import make_graph
from database import create_db_engine
from ingestion.vector_store import EMBEDDING_MODEL


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Build shared resources once at startup and clean up on shutdown.
    Stored in app.state so route handlers can access them.
    """
    engine = create_db_engine()
    session = Session(engine)
    config = GraphConfig()
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    app.state.session = session
    app.state.embeddings = embeddings
    app.state.config = config
    app.state.graph = make_graph(config, session)

    yield

    session.close()
    engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="NomadLATAM RAG API",
        description="Grounded Q&A for digital nomads in Latin America.",
        version="0.1.0",
        lifespan=lifespan,
    )

    from api.routes.chat import router as chat_router
    from api.routes.documents import router as documents_router

    app.include_router(chat_router)
    app.include_router(documents_router)

    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()