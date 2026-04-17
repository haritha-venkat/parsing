"""
Q&A engine backed by a LangGraph RAG workflow.
"""

from langchain_core.documents import Document

from src.agentic.agentic_rag import AgenticRAG
from src.logger.log_setup import LoggerFactory
from src.retriever.retriever import Retriever
from src.vectorstore.chroma_store import VectorStore

logger = LoggerFactory.get_logger(__name__)


class QAEngine:
    """Ask questions against indexed documents."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.agentic_rag = AgenticRAG(
            retriever=retriever,
            vector_store=VectorStore(),
        )

    def ask(self, query: str, verbose: bool = True) -> dict:
        """
        Ask a question using the agentic RAG pipeline.

        Returns:
            dict containing the query, route, retrieved documents, and answer.
        """
        logger.info("Question: %s", query)

        state = self.agentic_rag.invoke(query)
        results = state["documents"]

        if state["query_type"] == "document" and not results:
            logger.warning("No results found for query: '%s'", query)
            print("\n[WARN] No results found. Make sure documents are indexed first.\n")
            return state

        if verbose:
            self._print_results(
                query=query,
                answer=state["answer"],
                query_type=state["query_type"],
                top_score=state["top_score"],
                results=results if state["query_type"] == "document" else [],
            )

        return state

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
