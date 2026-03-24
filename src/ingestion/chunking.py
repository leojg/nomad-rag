from abc import ABC, abstractmethod
from typing import Iterable
from ingestion.models import ChunkMetadata
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.embeddings import Embeddings

class ChunkingStrategy(ABC):
    """Splits one document into many chunks; metadata is copied and refined per chunk."""


    @property
    @abstractmethod
    def id(self) -> str:
        """Stable id stored in ChunkMetadata.chunk_strategy (e.g. 'recursive', 'markdown_headers', 'semantic')."""

    @abstractmethod
    def _iter_chunks(self, text: str) -> Iterable[tuple[str, str | None]]:
        """Iterate over the chunks of `text`."""

    def chunk(self, text: str, metadata: ChunkMetadata) -> list[tuple[str, ChunkMetadata]]:
        """Split `text` into chunks, refining `metadata` for each chunk."""
        chunks = []
        for chunk, section in self._iter_chunks(text):
            if not chunk.strip() or len(chunk.strip()) < 50:
                continue
            chunks.append(
                (
                    chunk, 
                    metadata.model_copy(update={"section": section, "chunk_strategy": self.id})
                )
            )

        return chunks


class RecursiveCharacterTextSplitterStrategy(ChunkingStrategy):
    """Split text into chunks of a maximum number of characters."""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self._splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @property
    def id(self) -> str:
        return "recursive"

    def _iter_chunks(self, text: str) -> Iterable[tuple[str, str | None]]:
        for chunk in self._splitter.split_text(text):
            yield chunk, None

class MarkdownHeaderTextSplitterStrategy(ChunkingStrategy):
    """Split text into chunks based on markdown headers."""

    def __init__(self, headers_to_split_on: list[tuple[str, str]]):
        self._splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    
    @property
    def id(self) -> str:
        return "markdown_headers"
    
    def _iter_chunks(self, text: str) -> Iterable[tuple[str, str | None]]:
        for doc in self._splitter.split_text(text):
            section = (
                doc.metadata.get("Header 3")
                or doc.metadata.get("Header 2")
                or doc.metadata.get("Header 1")
            )
            yield doc.page_content, section
    
class SemanticChunkerStrategy(ChunkingStrategy):
    """Split text into chunks based on semantic similarity."""

    def __init__(
        self, 
        embeddings: Embeddings, 
        buffer_size: int = 1, 
        min_chunk_size: int = 100, 
        breakpoint_threshold_type: str = "percentile"
    ):
        self._splitter = SemanticChunker(
            embeddings=embeddings, 
            buffer_size=buffer_size, 
            min_chunk_size=min_chunk_size, 
            breakpoint_threshold_type=breakpoint_threshold_type,
        )
    
    @property
    def id(self) -> str:
        return "semantic"
    
    def _iter_chunks(self, text: str) -> Iterable[tuple[str, str | None]]:
        for chunk in self._splitter.split_text(text):
            yield chunk, None