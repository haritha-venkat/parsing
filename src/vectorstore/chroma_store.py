"""
src/vectorstore/chroma_store.py
─────────────────────────────────
Persistent ChromaDB vector store via LangChain's Chroma wrapper.
Handles upsert, search, deletion, and DB introspection.
"""

from typing import List, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

import config
from src.embedder.embedder import EmbeddingModel
from src.logger.log_setup import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class VectorStore:
    """
    Persistent ChromaDB store backed by LangChain's Chroma integration.

    Args:
        collection_name : ChromaDB collection name (default from config).
        persist_dir     : Directory to persist vectors (default from config).
    """

    def __init__(
        self,
        collection_name: str = config.COLLECTION_NAME,
        persist_dir: str = str(config.CHROMA_DIR),
    ) -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._embedding_model = EmbeddingModel.get_instance()
        self._store: Chroma | None = None
        logger.info(
            "VectorStore configured (collection='%s', dir='%s')",
            collection_name,
            persist_dir,
        )

    # ── Store access ───────────────────────────────────────────────────────────

    @property
    def store(self) -> Chroma:
        """Lazy-initialise the Chroma store on first access."""
        if self._store is None:
            self._store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self._embedding_model.model,
                persist_directory=self.persist_dir,
                collection_metadata={"hnsw:space": config.CHROMA_DISTANCE},
            )
            logger.info(
                "ChromaDB store opened. Documents in collection: %d",
                self.count(),
            )
        return self._store

    # ── Indexing ───────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add documents to the vector store.
        Generates IDs from source + chunk_index to prevent duplicates.

        Args:
            documents: List of LangChain Document chunks.

        Returns:
            int: Number of documents added.
        """
        if not documents:
            logger.warning("add_documents called with empty list.")
            return 0

        ids = [self._make_id(doc) for doc in documents]
        self.store.add_documents(documents=documents, ids=ids)
        logger.info("Added %d document(s). DB total: %d", len(documents), self.count())
        return len(documents)

    # ── Search ─────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = config.INITIAL_K,
    ) -> List[Document]:
        """
        Cosine similarity search — returns top-k matching documents.

        Args:
            query : Search query string.
            k     : Number of results to return.

        Returns:
            List of LangChain Documents sorted by relevance.
        """
        k = min(k, self.count())
        if k == 0:
            logger.warning("Vector store is empty.")
            return []
        results = self.store.similarity_search(query, k=k)
        logger.info(
            "Similarity search returned %d result(s) for query: '%s'",
            len(results),
            query,
        )
        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = config.INITIAL_K,
    ) -> List[Tuple[Document, float]]:
        """
        Cosine search returning (Document, score) tuples.

        Args:
            query : Search query string.
            k     : Number of results.

        Returns:
            List of (Document, distance_score) tuples.
        """
        k = min(k, self.count())
        if k == 0:
            return []
        return self.store.similarity_search_with_score(query, k=k)

    # ── Utilities ──────────────────────────────────────────────────────────────

    def count(self) -> int:
        """Return total number of documents in the collection."""
        try:
            return self.store._collection.count()
        except Exception:
            return 0

    def clear(self) -> None:
        """Delete all documents from the collection."""
        self.store.delete_collection()
        self._store = None
        logger.warning("ChromaDB collection '%s' cleared.", self.collection_name)

    def info(self) -> dict:
        """Return a summary dict of the vector store state."""
        return {
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
            "total_documents": self.count(),
        }

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _make_id(doc: Document) -> str:
        """Generate a deterministic ID from document metadata."""
        import hashlib

        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 0)
        chunk = doc.metadata.get("chunk_index", 0)
        raw = f"{source}_{page}_{chunk}_{doc.page_content[:40]}"
        return hashlib.md5(raw.encode()).hexdigest()
