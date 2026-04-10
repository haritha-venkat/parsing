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
    query_type: str
    top_score: float | None


class RAGGraph:
    """
    LangGraph pipeline:
      1. retrieve relevant chunks from ChromaDB
      2. classify whether the query belongs to indexed documents
      3. answer with document context or as a general Groq question
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
            "query_type": "",
            "top_score": None,
        }
        return self.graph.invoke(initial_state)

    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("classify_query", self._classify_query)
        workflow.add_node("generate_document_answer", self._generate_document_answer)
        workflow.add_node("generate_general_answer", self._generate_general_answer)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "classify_query")
        workflow.add_conditional_edges(
            "classify_query",
            self._route_after_classification,
            {
                "document": "generate_document_answer",
                "general": "generate_general_answer",
            },
        )
        workflow.add_edge("generate_document_answer", END)
        workflow.add_edge("generate_general_answer", END)

        return workflow.compile()

    def _retrieve(self, state: RAGState) -> dict:
        documents = self.retriever.retrieve(state["query"])
        logger.info("LangGraph retrieve node returned %d document(s)", len(documents))
        return {"documents": documents}

    def _classify_query(self, state: RAGState) -> dict:
        documents = state["documents"]
        if not documents:
            logger.info("Query classified as general because no documents were found.")
            return {"query_type": "general", "top_score": None}

        top_score = documents[0].metadata.get("rerank_score")
        if top_score is None:
            logger.info(
                "Query classified as document because documents were retrieved."
            )
            return {"query_type": "document", "top_score": None}

        query_type = (
            "document"
            if float(top_score) >= config.DOC_RELEVANCE_THRESHOLD
            else "general"
        )
        logger.info(
            "Query classified as %s (top_score=%s, threshold=%s)",
            query_type,
            top_score,
            config.DOC_RELEVANCE_THRESHOLD,
        )
        return {"query_type": query_type, "top_score": float(top_score)}

    @staticmethod
    def _route_after_classification(state: RAGState) -> str:
        return state["query_type"]

    def _generate_document_answer(self, state: RAGState) -> dict:
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

        return self._invoke_llm(messages, mode="document")

    def _generate_general_answer(self, state: RAGState) -> dict:
        messages = [
            SystemMessage(
                content=(
                    "You are a helpful general assistant. Answer directly and "
                    "concisely. Do not claim that the answer came from indexed "
                    "documents."
                )
            ),
            HumanMessage(content=state["query"]),
        ]
        return self._invoke_llm(messages, mode="general")

    def _invoke_llm(
        self,
        messages: list[SystemMessage | HumanMessage],
        mode: str,
    ) -> dict:
        try:
            response = self.llm.invoke(messages)
            answer = str(response.content).strip()
            logger.info("LangGraph %s answer node completed.", mode)
            return {"answer": answer}
        except RateLimitError as exc:
            logger.error(
                "Groq request failed because credits or rate limit are exhausted."
            )
            return {
                "answer": (
                    f"The {mode} route worked, but Groq could not generate an "
                    "answer because the API account has exhausted credits or reached "
                    f"its spending limit. Provider message: {exc}"
                )
            }
        except APIError as exc:
            logger.error("Groq request failed: %s", exc)
            return {
                "answer": (
                    f"The {mode} route worked, but Groq could not generate an "
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
