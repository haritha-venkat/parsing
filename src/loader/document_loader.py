"""
src/loader/document_loader.py
──────────────────────────────
Multi-format document loader.

Supported formats:
  - PDF  → parsed by Marker (converts to markdown for clean extraction)
  - CSV  → row-by-row via LangChain CSVLoader
  - TXT  → LangChain TextLoader
  - DOCX → LangChain Docx2txtLoader
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    TextLoader,
    Docx2txtLoader,
)

import config
from src.logger.log_setup import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class DocumentLoader:
    """
    Loads documents from various file formats into LangChain Document objects.

    Each Document has:
        - page_content : str  — the raw text
        - metadata     : dict — source, page, file_type
    """

    SUPPORTED = config.SUPPORTED_EXTENSIONS

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, file_path: str) -> List[Document]:
        """
        Load a single file and return a list of LangChain Documents.

        Args:
            file_path: Absolute or relative path to the file.

        Returns:
            List[Document]: One document per page (PDF) or per row (CSV).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
        """
        path = Path(file_path).resolve()
        self._validate(path)

        ext = path.suffix.lower()
        logger.info("Loading file: %s (type=%s)", path.name, ext)

        loaders = {
            ".pdf": self._load_pdf,
            ".csv": self._load_csv,
            ".txt": self._load_txt,
            ".docx": self._load_docx,
        }

        docs = loaders[ext](path)
        logger.info("Loaded %d document(s) from '%s'", len(docs), path.name)
        return docs

    def load_many(self, file_paths: List[str]) -> List[Document]:
        """
        Load multiple files and combine all documents.

        Args:
            file_paths: List of file paths.

        Returns:
            List[Document]: Combined documents from all files.
        """
        all_docs: List[Document] = []
        for fp in file_paths:
            try:
                all_docs.extend(self.load(fp))
            except (FileNotFoundError, ValueError) as exc:
                logger.error("Skipping '%s': %s", fp, exc)
        logger.info("Total documents loaded: %d", len(all_docs))
        return all_docs

    # ── Private loaders ────────────────────────────────────────────────────────

    def _load_pdf(self, path: Path) -> List[Document]:
        """Use Marker to convert PDF → markdown, then wrap as Documents."""
        try:
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.output import text_from_rendered
        except ImportError as exc:
            raise ImportError(
                "Marker not installed. Run: pip install marker-pdf"
            ) from exc

        converter = PdfConverter(artifact_dict=create_model_dict())
        rendered = converter(str(path))
        full_text, _, _ = text_from_rendered(rendered)

        # Split by page markers that Marker inserts (## Page N)
        pages = full_text.split("\n\n")
        docs = []
        for i, page_text in enumerate(pages, 1):
            text = page_text.strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": path.name, "page": i, "file_type": "pdf"},
                    )
                )
        return docs

    def _load_csv(self, path: Path) -> List[Document]:
        """Load CSV rows as individual documents."""
        loader = CSVLoader(file_path=str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "csv"
        return docs

    def _load_txt(self, path: Path) -> List[Document]:
        """Load plain text file, auto-detecting encoding (utf-8, utf-16, latin-1)."""
        for encoding in ("utf-8-sig", "utf-16", "latin-1"):
            try:
                loader = TextLoader(file_path=str(path), encoding=encoding)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["file_type"] = "txt"
                logger.info("Loaded '%s' with encoding: %s", path.name, encoding)
                return docs
            except (UnicodeDecodeError, RuntimeError):
                continue

        raise RuntimeError(
            f"Could not decode '{path.name}'. "
            "Try re-saving as UTF-8 in VS Code: "
            "bottom-right corner → click encoding → Save with Encoding → UTF-8"
        )

    def _load_docx(self, path: Path) -> List[Document]:
        """Load DOCX file as a single document."""
        loader = Docx2txtLoader(file_path=str(path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "docx"
        return docs

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in self.SUPPORTED:
            raise ValueError(
                f"Unsupported file type '{path.suffix}'. "
                f"Supported: {self.SUPPORTED}"
            )
