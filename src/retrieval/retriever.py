"""LangChain BaseRetriever adapter over hybrid search."""

from __future__ import annotations

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from retrieval.hybrid import hybrid_search


class NomadRetriever(BaseRetriever):
    """Wraps hybrid_search as a LangChain-compatible retriever for M3 chain wiring."""

    k: int
    embeddings: OpenAIEmbeddings
    session: Session
    filters: dict[str, str] | None = None

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        results = hybrid_search(
            query=query,
            k=self.k,
            embeddings=self.embeddings,
            session=self.session,
            filters=self.filters,
        )
        return [
            Document(
                page_content=record.text,
                metadata={
                    "source_file": record.source_file,
                    "document_type": record.document_type,
                    "country": record.country,
                    "city": record.city,
                    "section": record.section,
                    "chunk_strategy": record.chunk_strategy,
                },
            )
            for record in results
        ]