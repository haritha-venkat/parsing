"""
Q&A engine backed by a LangGraph RAG workflow.
"""

import datetime
import json
from pathlib import Path

from langchain_core.documents import Document

import config
from src.graph.rag_graph import RAGGraph, RAGState
from src.logger.log_setup import LoggerFactory
from src.retriever.retriever import Retriever

logger = LoggerFactory.get_logger(__name__)


class QAEngine:
    """
    Ask questions against indexed documents.
    Every Q&A pair is persisted to a dated JSON file in qa_history/.
    """

    def __init__(
        self,
        retriever: Retriever,
        history_dir: Path = config.QA_HISTORY_DIR,
    ) -> None:
        self.retriever = retriever
        self.history_dir = history_dir
        self._history_file = history_dir / f"qa_{datetime.date.today()}.json"
        self.graph = RAGGraph(retriever=retriever)

    def ask(self, query: str, verbose: bool = True) -> RAGState:
        """
        Ask a question using the LangGraph RAG pipeline.

        Returns:
            RAGState containing the query, route, retrieved documents, and Groq answer.
        """
        logger.info("Question: %s", query)

        state = self.graph.invoke(query)
        results = state["documents"]

        if state["query_type"] == "document" and not results:
            logger.warning("No results found for query: '%s'", query)
            print("\n[WARN] No results found. Make sure documents are indexed first.\n")
            return state

        self._save_to_history(
            query=query,
            answer=state["answer"],
            query_type=state["query_type"],
            top_score=state["top_score"],
            results=results if state["query_type"] == "document" else [],
        )

        if verbose:
            self._print_results(
                query=query,
                answer=state["answer"],
                query_type=state["query_type"],
                top_score=state["top_score"],
                results=results if state["query_type"] == "document" else [],
            )

        return state

    def load_history(self) -> list[dict]:
        """Load today's Q&A history from JSON."""
        if not self._history_file.exists():
            return []
        with open(self._history_file, encoding="utf-8") as fh:
            return json.load(fh)

    def print_history(self, last_n: int | None = None) -> None:
        """Print Q&A history to stdout."""
        history = self.load_history()
        if not history:
            print("No Q&A history found for today.")
            return

        entries = history[-last_n:] if last_n else history
        print(f"\nQ&A History ({len(entries)} of {len(history)} entries)\n")

        for i, record in enumerate(entries, 1):
            print(f"[{i}] {record['timestamp']}")
            print(f"    Question: {record['question']}")
            print(f"    Route: {record.get('query_type', 'document')}")
            print(f"    Answer: {record.get('answer', 'N/A')}")
            for ans in record["answers"]:
                score = ans.get("rerank_score", "N/A")
                print(
                    f"    Source #{ans['rank']} | {ans['source']} "
                    f"(pg {ans['page']}) | score: {score}"
                )
            print()

    def _save_to_history(
        self,
        query: str,
        answer: str,
        query_type: str,
        top_score: float | None,
        results: list[Document],
    ) -> None:
        """Append a Q&A record to today's JSON history file."""
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": query,
            "answer": answer,
            "query_type": query_type,
            "top_score": top_score,
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
            "Q&A saved to history (total today: %d) -> %s",
            len(history),
            self._history_file.name,
        )

    @staticmethod
    def _print_results(
        query: str,
        answer: str,
        query_type: str,
        top_score: float | None,
        results: list[Document],
    ) -> None:
        """Pretty-print generated answer and source chunks."""
        sep = "=" * 65
        print(f"\n{sep}")
        print(f"Question: {query}")
        print(f"Route: {query_type} | top_score: {top_score}")
        print(sep)
        print("\nAnswer")
        print("-" * 65)
        print(answer)
        if not results:
            print("\nSources: not used for general-answer route")
            print(f"\n{sep}\n")
            return

        print("\nSources")

        for i, doc in enumerate(results, 1):
            score = doc.metadata.get("rerank_score", "N/A")
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            print(f"\nResult #{i} | {source} (pg {page}) | score: {score}")
            print("-" * 65)
            snippet = doc.page_content[:600]
            if len(doc.page_content) > 600:
                snippet += "..."
            print(snippet)
        print(f"\n{sep}\n")
