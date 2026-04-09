import argparse
import sys

import config  # noqa: F401 — ensures dirs are created on import
from src.logger.log_setup import LoggerFactory
from src.loader.document_loader import DocumentLoader
from src.chunker.text_chunker import TextChunker
from src.vectorstore.chroma_store import VectorStore
from src.reranker.reranker import Reranker
from src.retriever.retriever import Retriever
from src.qa.qa_engine import QAEngine

logger = LoggerFactory.get_logger(__name__)


# ── Pipeline assembly ──────────────────────────────────────────────────────────


def build_pipeline(use_rerank: bool = True) -> QAEngine:
    """
    Assemble and return the full RAG pipeline.

    Returns:
        QAEngine: Ready-to-use Q&A engine.
    """
    vector_store = VectorStore()
    reranker = Reranker()
    retriever = Retriever(
        vector_store=vector_store,
        reranker=reranker,
        use_rerank=use_rerank,
    )
    return QAEngine(retriever=retriever)


# ── CLI commands ───────────────────────────────────────────────────────────────


def cmd_index(args: argparse.Namespace) -> None:
    """Load, chunk, embed and store documents."""
    loader = DocumentLoader()
    chunker = TextChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    vector_store = VectorStore()

    logger.info("Indexing %d file(s)...", len(args.files))
    docs = loader.load_many(args.files)
    chunks = chunker.split(docs)
    added = vector_store.add_documents(chunks)

    print(f"\n✅ Indexed {added} chunks from {len(args.files)} file(s).")
    print(f"   Total docs in DB: {vector_store.count()}\n")


def cmd_ask(args: argparse.Namespace) -> None:
    """Ask a question against the indexed documents."""
    engine = build_pipeline(use_rerank=not args.no_rerank)
    engine.ask(args.query)


def cmd_history(args: argparse.Namespace) -> None:
    """Print today's Q&A history."""
    engine = build_pipeline()
    engine.print_history(last_n=args.last_n)


def cmd_info(_args: argparse.Namespace) -> None:
    """Print vector store statistics."""
    vector_store = VectorStore()
    info = vector_store.info()
    print("\n📦 ChromaDB Info")
    for key, val in info.items():
        print(f"   {key:<20}: {val}")
    print()


def cmd_clear(_args: argparse.Namespace) -> None:
    """Clear all documents from the vector store."""
    confirm = input("⚠️  This will delete all indexed data. Type 'yes' to confirm: ")
    if confirm.strip().lower() == "yes":
        VectorStore().clear()
        print("✅ Vector store cleared.")
    else:
        print("Cancelled.")


# ── Argument parser ────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="RAG Pipeline — ML Team Task",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index documents into ChromaDB")
    p_index.add_argument(
        "--files", nargs="+", required=True, help="File paths to index"
    )
    p_index.add_argument("--chunk-size", type=int, default=config.CHUNK_SIZE)
    p_index.add_argument("--chunk-overlap", type=int, default=config.CHUNK_OVERLAP)

    # ask
    p_ask = sub.add_parser("ask", help="Ask a question")
    p_ask.add_argument("--query", required=True, help="Your question")
    p_ask.add_argument("--no-rerank", action="store_true", help="Skip re-ranking")

    # history
    p_hist = sub.add_parser("history", help="Show Q&A history")
    p_hist.add_argument("--last-n", type=int, default=None)

    # info
    sub.add_parser("info", help="Show vector store info")

    # clear
    sub.add_parser("clear", help="Clear the vector store")

    return parser


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "index": cmd_index,
        "ask": cmd_ask,
        "history": cmd_history,
        "info": cmd_info,
        "clear": cmd_clear,
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
