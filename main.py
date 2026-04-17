import argparse
import os
import re as _re
import sys
from pathlib import Path

import config
from src.chunker.text_chunker import TextChunker
from src.loader.capex_loader import load_capex_batch
from src.logger.log_setup import LoggerFactory
from src.qa.qa_engine import QAEngine
from src.reranker.reranker import Reranker
from src.retriever.retriever import Retriever
from src.vectorstore.chroma_store import VectorStore

logger = LoggerFactory.get_logger(__name__)

# ── LangSmith tracing setup ────────────────────────────────────────────────────
if config.LANGSMITH_API_KEY:
    os.environ["LANGCHAIN_TRACING_V2"] = config.LANGSMITH_TRACING
    os.environ["LANGCHAIN_API_KEY"] = config.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = config.LANGSMITH_PROJECT

# ── Pipeline assembly ──────────────────────────────────────────────────────────


def build_pipeline(use_rerank: bool = True) -> QAEngine:
    """Assemble and return the full RAG pipeline."""
    vector_store = VectorStore()
    reranker = Reranker()
    retriever = Retriever(
        vector_store=vector_store,
        reranker=reranker,
        use_rerank=use_rerank,
    )
    return QAEngine(retriever=retriever)


# ── CLI commands ───────────────────────────────────────────────────────────────


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question against the indexed capex documents."""
    engine = build_pipeline(use_rerank=not args.no_rerank)
    engine.ask(args.query)


def cmd_info(_args: argparse.Namespace) -> None:
    """Print vector store statistics."""
    vector_store = VectorStore()
    info = vector_store.info()
    print("\nChromaDB Info")
    for key, val in info.items():
        print(f"   {key:<20}: {val}")
    print()


# ── Mining PDF auto-discovery ──────────────────────────────────────────────────
_DEFAULT_PDF_DIR = config.MINING_PDF_DIR


def _company_name_from_filename(stem: str) -> str:
    """Derive a readable company name from a PDF filename stem."""
    name = stem.replace("_", " ").replace("-", " ")
    return _re.sub(r"\s+", " ", name).strip()


def _discover_targets(pdf_dir: Path) -> dict[str, Path]:
    """Return {company_name: pdf_path} for every PDF found in pdf_dir."""
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    return {_company_name_from_filename(p.stem): p for p in pdfs}


def cmd_ingest_capex(args: argparse.Namespace) -> None:
    """Extract capex data from mining PDFs and index into ChromaDB."""
    pdf_dir = Path(args.dir)

    if not pdf_dir.exists():
        print(f"[ERROR] Directory not found: {pdf_dir}")
        return

    if args.files:
        targets: dict[str, Path] = {}
        for item in args.files:
            if "=" not in item:
                print(
                    f"[WARN] Skipping '{item}' — expected format 'CompanyName=path/to.pdf'"
                )
                continue
            company, pdf_path = item.split("=", 1)
            targets[company.strip()] = Path(pdf_path.strip())
    else:
        targets = _discover_targets(pdf_dir)

    if not targets:
        print(f"[WARN] No PDF files found in: {pdf_dir}")
        return

    print(f"\nIngesting capex data from {len(targets)} PDF(s) in '{pdf_dir.name}/'...")
    for company, path in targets.items():
        status = "OK     " if path.exists() else "MISSING"
        print(f"  [{status}] {path.name}  ->  {company}")

    docs = load_capex_batch(targets, context_lines=args.context_lines)

    if not docs:
        print("\n[WARN] No capex documents extracted. Check PDF paths above.")
        return

    chunker = TextChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    chunks = chunker.split(docs)

    vector_store = VectorStore()
    added = vector_store.add_documents(chunks)

    print(f"\nIndexed {added} capex chunks from {len(targets)} PDF(s).")
    print(f"Total docs in DB: {vector_store.count()}")
    print("\nExample queries:")
    print(
        '  python main.py ask --query "What was Barrick Gold capital expenditure in 2022?"'
    )
    print(
        '  python main.py ask --query "Compare sustaining vs growth capex across all companies"'
    )
    print(
        '  python main.py ask --query "Which mine had the highest development capital?"'
    )
    print()


def cmd_clear(_args: argparse.Namespace) -> None:
    """Clear all documents from the vector store."""
    confirm = input("This will delete all indexed data. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        VectorStore().clear()
        print("Vector store cleared.")
    else:
        print("Cancelled.")


# ── Argument parser ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Mining Capex RAG Pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question about mining capex")
    p_ask.add_argument("--query", required=True, help="Your question")
    p_ask.add_argument("--no-rerank", action="store_true", help="Skip re-ranking")

    # info
    sub.add_parser("info", help="Show vector store info")

    # clear
    sub.add_parser("clear", help="Clear the vector store")

    # ingest-capex
    p_capex = sub.add_parser(
        "ingest-capex",
        help="Extract capex data from mining PDFs and index into ChromaDB",
    )
    p_capex.add_argument(
        "--dir",
        default=str(_DEFAULT_PDF_DIR),
        metavar="FOLDER",
        help="Folder containing mining PDFs (default: data/)",
    )
    p_capex.add_argument(
        "--files",
        nargs="*",
        metavar="COMPANY=PATH",
        help="Optional overrides: 'CompanyName=path/to.pdf'",
    )
    p_capex.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    p_capex.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)
    p_capex.add_argument(
        "--context-lines",
        type=int,
        default=5,
        help="Lines of context around each capex keyword hit (default: 5)",
    )

    return parser


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "ask": cmd_ask,
        "info": cmd_info,
        "clear": cmd_clear,
        "ingest-capex": cmd_ingest_capex,
    }

    try:
        commands[args.command](args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Unhandled error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
