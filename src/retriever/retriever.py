"""
src/retriever/retriever.py
───────────────────────────
Orchestrates retrieval:
  1. Query → ChromaDB (initial_k candidates)
  2. Candidates → CrossEncoder re-ranker (final_k results)
"""

from typing import List

from langchain_core.documents import Document

import config
from src.vectorstore.chroma_store import VectorStore
from src.reranker.reranker import Reranker
from src.logger.log_setup import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class Retriever:
    """
    Two-stage retriever: vector search → re-ranking.

    Args:
        vector_store : Initialised VectorStore instance.
        reranker     : Initialised Reranker instance.
        initial_k    : Candidates fetched from ChromaDB.
        final_k      : Results returned after re-ranking.
        use_rerank   : Toggle re-ranking on/off.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        reranker: Reranker,
        initial_k: int = config.INITIAL_K,
        final_k: int = config.FINAL_K,
        use_rerank: bool = True,
    ) -> None:
        self.vector_store = vector_store
        self.reranker = reranker
        self.initial_k = initial_k
        self.final_k = final_k
        self.use_rerank = use_rerank

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve the most relevant documents for a query.

        Args:
            query: User's search query.

        Returns:
            List[Document]: Top documents with 'rerank_score' in metadata
                            (if re-ranking is enabled).
        """
        logger.info("Retrieving for query: '%s'", query)

        # Stage 1 — Vector search
        candidates = self.vector_store.similarity_search(query, k=self.initial_k)
        logger.info("Stage 1: %d candidate(s) from vector search", len(candidates))

        if not candidates:
            return []

        # Stage 2 — Re-rank
        if self.use_rerank:
            results = self.reranker.rerank(query, candidates, top_k=self.final_k)
        else:
            results = candidates[: self.final_k]

        logger.info("Final results: %d document(s) returned", len(results))
        return results
