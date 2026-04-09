"""
src/reranker/reranker.py
─────────────────────────
CrossEncoder-based re-ranker.
Takes initial vector-search candidates and re-scores them
for higher precision before returning final results.
"""

from typing import List

from langchain_core.documents import Document
from sentence_transformers.cross_encoder import CrossEncoder

import config
from src.logger.log_setup import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class Reranker:
    """
    Re-ranks retrieved documents using a CrossEncoder model.

    CrossEncoder performs full attention between query + document,
    giving more precise relevance scores than bi-encoder cosine search.

    Args:
        model_name : CrossEncoder HuggingFace model (default from config).
    """

    def __init__(self, model_name: str = config.RERANK_MODEL_NAME) -> None:
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        logger.info("Reranker configured with model: %s", model_name)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the CrossEncoder on first use."""
        if self._model is None:
            logger.info("Loading CrossEncoder: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("CrossEncoder loaded.")
        return self._model

    # ── Public API ─────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int = config.FINAL_K,
    ) -> List[Document]:
        """
        Re-rank documents by relevance to the query.

        Args:
            query     : The user's search query.
            documents : Candidate documents from initial retrieval.
            top_k     : Number of documents to return after re-ranking.

        Returns:
            List[Document]: Top-k documents sorted by re-rank score (desc).
                            Each document has 'rerank_score' added to metadata.
        """
        if not documents:
            logger.warning("Reranker received empty document list.")
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores: List[float] = self.model.predict(pairs).tolist()

        for doc, score in zip(documents, scores):
            doc.metadata["rerank_score"] = round(float(score), 6)

        ranked = sorted(
            documents, key=lambda d: d.metadata["rerank_score"], reverse=True
        )

        logger.info(
            "Re-ranked %d candidate(s) → returning top %d",
            len(documents),
            top_k,
        )
        return ranked[:top_k]
