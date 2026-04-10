"""
LangGraph workflow for retrieval-augmented question answering.
"""

from typing import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import APIError, RateLimitError

import config
from src.logger.log_setup import LoggerFactory
from src.retriever.retriever import Retriever

logger = LoggerFactory.get_logger(__name__)


class RAGState(TypedDict):
    """State passed between LangGraph nodes."""

    query: str
    documents: list[Document]
    answer: str


class RAGGraph:
    """
    LangGraph pipeline:
      1. retrieve relevant chunks from ChromaDB
      2. generate a final answer using Groq
    """

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self._llm: ChatOpenAI | None = None
        self.graph = self._build_graph()

    @property
    def llm(self) -> ChatOpenAI:
        """Create the Groq client lazily so missing keys fail only on ask."""
        if self._llm is None:
            if not config.GROQ_API_KEY:
                raise ValueError(
                    "Missing GROQ_API_KEY. Add it to a .env file or set it in your "
                    "terminal before running the ask command."
                )

            self._llm = ChatOpenAI(
                model=config.GROQ_MODEL_NAME,
                api_key=config.GROQ_API_KEY,
                base_url=config.GROQ_BASE_URL,
                temperature=config.LLM_TEMPERATURE,
            )
        return self._llm

    def invoke(self, query: str) -> RAGState:
        """Run the compiled LangGraph workflow for one user query."""
        initial_state: RAGState = {
            "query": query,
            "documents": [],
            "answer": "",
        }
        return self.graph.invoke(initial_state)

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("generate_answer", self._generate_answer)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate_answer")
        workflow.add_edge("generate_answer", END)

        return workflow.compile()

    def _retrieve(self, state: RAGState) -> dict:
        documents = self.retriever.retrieve(state["query"])
        logger.info("LangGraph retrieve node returned %d document(s)", len(documents))
        return {"documents": documents}

    def _generate_answer(self, state: RAGState) -> dict:
        documents = state["documents"]
        if not documents:
            return {
                "answer": "No relevant documents were found. Please index documents first."
            }

        context = self._format_context(documents)
        messages = [
            SystemMessage(
                content=(
                    "You are a RAG assistant. Answer only from the provided context. "
                    "If the context does not contain the answer, say that the indexed "
                    "documents do not provide enough information."
                )
            ),
            HumanMessage(
                content=(
                    f"Question:\n{state['query']}\n\n"
                    f"Context:\n{context}\n\n"
                    "Answer clearly and cite the source file/page when available."
                )
            ),
        ]

        try:
            response = self.llm.invoke(messages)
            answer = str(response.content).strip()
            logger.info("LangGraph generate_answer node completed.")
            return {"answer": answer}
        except RateLimitError as exc:
            logger.error(
                "Groq request failed because credits or rate limit are exhausted."
            )
            return {
                "answer": (
                    "The retrieval step worked, but Groq could not generate an "
                    "answer because the API account has exhausted credits or reached "
                    f"its spending limit. Provider message: {exc}"
                )
            }
        except APIError as exc:
            logger.error("Groq request failed: %s", exc)
            return {
                "answer": (
                    "The retrieval step worked, but Groq could not generate an "
                    f"answer because the API request failed. Provider message: {exc}"
                )
            }

    @staticmethod
    def _format_context(documents: list[Document]) -> str:
        sections = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            sections.append(f"[{i}] Source: {source}, page: {page}\n{doc.page_content}")
        return "\n\n".join(sections)
