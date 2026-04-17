"""
config.py
─────────
Central configuration for the RAG project.
All paths, model names, and tunable parameters live here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Project root & directories ─────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DIR: Path = BASE_DIR / "chroma_db"
LOG_DIR: Path = BASE_DIR / "logs"
# Auto-create required directories
for _dir in (DATA_DIR, CHROMA_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL: str = "INFO"

# ── Embedding model ────────────────────────────────────────────────────────────
EMBED_MODEL_NAME: str = "all-MiniLM-L6-v2"
EMBED_DEVICE: str = "cpu"  # change to "cuda" if GPU available

# ── ChromaDB ───────────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "rag_collection"
CHROMA_DISTANCE: str = "cosine"

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 100

# ── Retrieval ──────────────────────────────────────────────────────────────────
INITIAL_K: int = 10
FINAL_K: int = 5
DOC_RELEVANCE_THRESHOLD: float = float(os.getenv("DOC_RELEVANCE_THRESHOLD", "-11.0"))

# ── Re-ranking ─────────────────────────────────────────────────────────────────
RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Groq LLM
GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL_NAME: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# ── Supported file types ───────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: tuple = (".pdf", ".csv", ".txt", ".docx")

# ── Mining PDF source directory ────────────────────────────────────────────────
MINING_PDF_DIR: Path = Path(os.getenv("MINING_PDF_DIR", str(DATA_DIR)))


# ── LangSmith ──────────────────────────────────────────────────────────────────
LANGSMITH_API_KEY: str | None = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "rag-project")
