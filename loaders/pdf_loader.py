"""
loaders/pdf_loader.py
---------------------
PDF loader powered by **Marker** (marker-pdf).
Marker converts PDFs to clean Markdown, preserving headings, tables,
and layout — far better than raw text extraction.
"""

from pathlib import Path

from loaders.base_loader import BaseLoader
from logger.logger import get_logger

logger = get_logger(__name__)


class PDFLoader(BaseLoader):
    """
    Load PDF files via the Marker library.

    Marker runs a layout-aware ML pipeline that outputs Markdown text,
    which is then stored as the ``text`` field of each document dict.
    Each logical page from the PDF becomes one document.

    Attributes:
        _models: Cached Marker model bundle (loaded once per instance).
    """

    def __init__(self) -> None:
        """Initialise the PDF loader and load Marker models."""
        logger.info("Loading Marker models for PDF parsing …")
        try:
            from marker.models import create_model_dict

            self._models = create_model_dict()
            logger.info("Marker models loaded successfully.")
        except ImportError as exc:
            raise ImportError(
                "marker-pdf is not installed. Run: pip install marker-pdf"
            ) from exc

    @property
    def supported_extensions(self) -> tuple[str, ...]:
        return (".pdf",)

    def load(self, file_path: str | Path) -> list[dict]:
        """
        Convert a PDF to Markdown and return one document per page.

        Args:
            file_path: Path to the PDF file.

        Returns:
            list[dict]: One dict per page with keys
                ``source``, ``page``, ``text``, ``metadata``.
        """
        path = Path(file_path)
        self._validate(path)
        logger.info("Parsing PDF with Marker: %s", path.name)

        try:
            from marker.converters.pdf import PdfConverter
            from marker.output import text_from_rendered

            converter = PdfConverter(artifact_dict=self._models)
            rendered = converter(str(path))
            full_markdown, _, metadata = text_from_rendered(rendered)
        except Exception as exc:
            logger.error("Marker failed on '%s': %s", path.name, exc)
            raise

        # Split into pages using Marker's page separator
        raw_pages = full_markdown.split("\f")  # form-feed page breaks
        if len(raw_pages) == 1:
            # Fallback: treat entire output as one page
            raw_pages = [full_markdown]

        docs = []
        for page_num, page_text in enumerate(raw_pages, start=1):
            cleaned = page_text.strip()
            if cleaned:
                docs.append(
                    {
                        "source": path.name,
                        "page": page_num,
                        "text": cleaned,
                        "metadata": {
                            "file_path": str(path),
                            "total_pages": len(raw_pages),
                        },
                    }
                )

        logger.info("PDF '%s' → %d page(s) extracted.", path.name, len(docs))
        return docs
