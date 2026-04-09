"""
src/embedder/embedder.py
─────────────────────────
Wraps LangChain's HuggingFaceEmbeddings (sentence-transformers).
Provides a single interface for encoding text.
"""

from typing import List

from langchain_huggingface import HuggingFaceEmbeddings

import config
from src.logger.log_setup import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class EmbeddingModel:
    """
    Singleton-style wrapper around LangChain HuggingFaceEmbeddings.

    Loads the model once on first access to avoid repeated I/O.

    Args:
        model_name : HuggingFace model name (default from config).
        device     : 'cpu' or 'cuda' (default from config).
    """

    _instance: "EmbeddingModel | None" = None

    def __init__(
        self,
        model_name: str = config.EMBED_MODEL_NAME,
        device: str = config.EMBED_DEVICE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: HuggingFaceEmbeddings | None = None

    # ── Singleton factory ──────────────────────────────────────────────────────

    @classmethod
    def get_instance(cls) -> "EmbeddingModel":
        """Return (or create) the shared EmbeddingModel instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """Lazy-load the embedding model on first access."""
        if self._model is None:
            logger.info(
                "Loading embedding model: %s on %s", self.model_name, self.device
            )
            self._model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": self.device},
                encode_kwargs={"normalize_embeddings": True},
            )
            logger.info("Embedding model loaded.")
        return self._model

    # ── Public API ─────────────────────────────────────────────────────────────

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of strings.

        Args:
            texts: List of text strings.

        Returns:
            List of embedding vectors.
        """
        return self.model.embed_documents(texts)

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query string.

        Args:
            query: Query text.

        Returns:
            Single embedding vector.
        """
        return self.model.embed_query(query)
