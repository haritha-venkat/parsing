"""
src/chunker/text_chunker.py
────────────────────────────
Splits LangChain Documents into smaller overlapping chunks
using LangChain's RecursiveCharacterTextSplitter.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from src.logger.log_setup import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class TextChunker:
    """
    Wraps LangChain's RecursiveCharacterTextSplitter with project defaults.

    The splitter tries to split on paragraphs → sentences → words → chars
    in that priority order, preserving semantic boundaries.

    Args:
        chunk_size    : Max characters per chunk (default from config).
        chunk_overlap : Characters shared between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
            add_start_index=True,  # adds 'start_index' to metadata
        )

        logger.info(
            "TextChunker ready (chunk_size=%d, chunk_overlap=%d)",
            chunk_size,
            chunk_overlap,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def split(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of Documents into smaller chunks.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List[Document]: Chunked documents with updated metadata.
        """
        if not documents:
            logger.warning("No documents to chunk.")
            return []

        chunks = self._splitter.split_documents(documents)
        self._add_chunk_indices(chunks)

        logger.info("Chunked %d document(s) → %d chunk(s)", len(documents), len(chunks))
        return chunks

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _add_chunk_indices(chunks: List[Document]) -> None:
        """
        Group chunks by source and add a sequential chunk_index per source.
        Mutates the list in-place.
        """
        source_counter: dict = {}
        for chunk in chunks:
            src = chunk.metadata.get("source", "unknown")
            source_counter[src] = source_counter.get(src, -1) + 1
            chunk.metadata["chunk_index"] = source_counter[src]
