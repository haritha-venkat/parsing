"""
src/qa/qa_engine.py
────────────────────
Q&A engine: retrieves relevant chunks and saves every
question + answer to a dated JSON file.
"""

import json
import datetime
from pathlib import Path
from typing import List

from langchain_core.documents import Document

import config
from src.retriever.retriever import Retriever
from src.logger.log_setup import LoggerFactory

logger = LoggerFactory.get_logger(__name__)


class QAEngine:
    """
    Ask questions against indexed documents.
    Every Q&A pair is persisted to a dated JSON file in qa_history/.

    Args:
        retriever       : Initialised Retriever instance.
        history_dir     : Directory to save JSON history files.
    """

    def __init__(
        self,
        retriever: Retriever,
        history_dir: Path = config.QA_HISTORY_DIR,
    ) -> None:
        self.retriever = retriever
        self.history_dir = history_dir
        self._history_file = history_dir / f"qa_{datetime.date.today()}.json"

    # ── Public API ─────────────────────────────────────────────────────────────

    def ask(self, query: str, verbose: bool = True) -> List[Document]:
        """
        Ask a question and retrieve relevant document chunks.

        Args:
            query   : User's natural language question.
            verbose : Print formatted results to stdout.

        Returns:
            List[Document]: Retrieved (and re-ranked) documents.
        """
        logger.info("Question: %s", query)

        results = self.retriever.retrieve(query)

        if not results:
            logger.warning("No results found for query: '%s'", query)
            print("\n⚠️  No results found. Make sure documents are indexed first.\n")
            return []

        self._save_to_history(query, results)

        if verbose:
            self._print_results(query, results)

        return results

    def load_history(self) -> List[dict]:
        """
        Load today's Q&A history from JSON.

        Returns:
            List of Q&A records for today.
        """
        if not self._history_file.exists():
            return []
        with open(self._history_file, encoding="utf-8") as fh:
            return json.load(fh)

    def print_history(self, last_n: int | None = None) -> None:
        """
        Print Q&A history to stdout.

        Args:
            last_n: Show only the last N entries (None = all).
        """
        history = self.load_history()
        if not history:
            print("No Q&A history found for today.")
            return

        entries = history[-last_n:] if last_n else history
        print(f"\n📜 Q&A History ({len(entries)} of {len(history)} entries)\n")

        for i, record in enumerate(entries, 1):
            print(f"[{i}] {record['timestamp']}")
            print(f"    ❓  {record['question']}")
            for ans in record["answers"]:
                score = ans.get("rerank_score", "N/A")
                print(
                    f"    📄  #{ans['rank']} | {ans['source']} "
                    f"(pg {ans['page']}) | score: {score}"
                )
            print()

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _save_to_history(self, query: str, results: List[Document]) -> None:
        """Append a Q&A record to today's JSON history file."""
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": query,
            "answers": [
                {
                    "rank": i + 1,
                    "text": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "rerank_score": doc.metadata.get("rerank_score"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                }
                for i, doc in enumerate(results)
            ],
        }

        history = self.load_history()
        history.append(record)

        with open(self._history_file, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2, ensure_ascii=False)

        logger.info(
            "Q&A saved to history (total today: %d) → %s",
            len(history),
            self._history_file.name,
        )

    @staticmethod
    def _print_results(query: str, results: List[Document]) -> None:
        """Pretty-print retrieval results."""
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"❓  {query}")
        print(sep)
        for i, doc in enumerate(results, 1):
            score = doc.metadata.get("rerank_score", "N/A")
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"\n📄  Result #{i}  |  {source}  (pg {page})  |  score: {score}")
            print("─" * 65)
            snippet = doc.page_content[:600]
            if len(doc.page_content) > 600:
                snippet += "..."
            print(snippet)
        print(f"\n{sep}\n")
