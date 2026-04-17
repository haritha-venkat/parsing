"""
src/loader/capex_loader.py
───────────────────────────
PyMuPDF-based loader that extracts capital expenditure text from mining
company PDFs and returns them as LangChain Documents ready for indexing.

Each Document carries metadata:
    source        : PDF filename
    company       : Mining company name
    page          : 1-indexed page number
    amounts_found : Comma-separated dollar amounts extracted from the snippet
    file_type     : "capex_pdf"
    chunk_index   : Sequential index per source file
"""

import re
from pathlib import Path

import fitz  # PyMuPDF
from langchain_core.documents import Document

from src.logger.log_setup import LoggerFactory

logger = LoggerFactory.get_logger(__name__)

# ── Capex signal keywords ──────────────────────────────────────────────────────
CAPEX_KEYWORDS = [
    r"capital expenditure",
    r"capital spending",
    r"capex",
    r"capital investment",
    r"sustaining capital",
    r"growth capital",
    r"expansionary capital",
    r"development capital",
    r"total capital",
]

# Regex to capture dollar amounts  e.g. $1,234.5M  /  US$456 million  /  $2.1B
AMOUNT_RE = re.compile(
    r"(?:US\$|USD|\$)?\s*[\d,]+(?:\.\d+)?\s*(?:million|billion|M\b|B\b|mn\b)",
    re.IGNORECASE,
)


class CapexLoader:
    """
    Extract capex-related text from a mining PDF using PyMuPDF.

    Args:
        company_name  : Human-readable company name stored in metadata.
        context_lines : Lines of context captured above/below a capex hit.
    """

    def __init__(self, company_name: str, context_lines: int = 5) -> None:
        self.company_name = company_name
        self.context_lines = context_lines

    # ── Public API ─────────────────────────────────────────────────────────────

    def load(self, pdf_path: str | Path) -> list[Document]:
        """
        Open *pdf_path*, scan every page for capex keywords, and return
        one LangChain Document per unique text snippet found.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of LangChain Documents with capex text and metadata.

        Raises:
            FileNotFoundError: If the PDF does not exist.
        """
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        logger.info(
            "CapexLoader scanning '%s' for company '%s'",
            path.name,
            self.company_name,
        )

        doc = fitz.open(str(path))
        documents: list[Document] = []
        seen: set[str] = set()
        chunk_idx = 0

        for page_num in range(len(doc)):
            page_text = doc[page_num].get_text("text")
            page_lower = page_text.lower()

            # Skip pages with no capex signals
            if not any(re.search(kw, page_lower) for kw in CAPEX_KEYWORDS):
                continue

            snippets = self._extract_snippets(page_text)
            for snippet in snippets:
                dedup_key = snippet[:100]
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)

                amounts = AMOUNT_RE.findall(snippet)
                documents.append(
                    Document(
                        page_content=snippet,
                        metadata={
                            "source": path.name,
                            "company": self.company_name,
                            "page": page_num + 1,
                            "amounts_found": ", ".join(amounts) if amounts else "",
                            "file_type": "capex_pdf",
                            "chunk_index": chunk_idx,
                        },
                    )
                )
                chunk_idx += 1

        doc.close()
        logger.info(
            "CapexLoader: %d capex chunk(s) extracted from '%s'",
            len(documents),
            path.name,
        )
        return documents

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _extract_snippets(self, page_text: str) -> list[str]:
        """
        Find every line on the page that matches a capex keyword and return
        it together with surrounding lines for context.
        """
        lines = page_text.splitlines()
        snippets: list[str] = []

        for i, line in enumerate(lines):
            if not any(re.search(kw, line, re.IGNORECASE) for kw in CAPEX_KEYWORDS):
                continue
            start = max(0, i - 1)
            end = min(len(lines), i + self.context_lines + 1)
            snippet = "\n".join(lines[start:end]).strip()
            if snippet:
                snippets.append(snippet)

        return snippets


# ── Convenience: load a batch of (company, path) pairs ────────────────────────


def load_capex_batch(
    targets: dict[str, str | Path],
    context_lines: int = 5,
) -> list[Document]:
    """
    Load capex documents from multiple PDFs.

    Args:
        targets       : Mapping of {company_name: pdf_path}.
        context_lines : Lines of context per snippet.

    Returns:
        Combined list of LangChain Documents from all PDFs.
    """
    all_docs: list[Document] = []
    for company, pdf_path in targets.items():
        loader = CapexLoader(company_name=company, context_lines=context_lines)
        try:
            docs = loader.load(pdf_path)
            all_docs.extend(docs)
        except FileNotFoundError as exc:
            logger.error("Skipping '%s': %s", company, exc)
    logger.info("Capex batch total: %d document(s) loaded", len(all_docs))
    return all_docs
