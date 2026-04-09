"""
config.py
─────────
Central configuration for the RAG project.
All paths, model names, and tunable parameters live here.
"""

from pathlib import Path

# ── Project root & directories ─────────────────────────────────────────────────
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
CHROMA_DIR: Path = BASE_DIR / "chroma_db"
LOG_DIR: Path = BASE_DIR / "logs"
QA_HISTORY_DIR: Path = BASE_DIR / "qa_history"

# Auto-create required directories
for _dir in (DATA_DIR, CHROMA_DIR, LOG_DIR, QA_HISTORY_DIR):
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
FINAL_K: int = 3

# ── Re-ranking ─────────────────────────────────────────────────────────────────
RERANK_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── Supported file types ───────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS: tuple = (".pdf", ".csv", ".txt", ".docx")
