"""
Agentic RAG layer built with a tool-calling agent on top of LangGraph.
"""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

import config
from src.logger.log_setup import LoggerFactory
from src.retriever.retriever import Retriever
from src.vectorstore.chroma_store import VectorStore

logger = LoggerFactory.get_logger(__name__)


class AgenticRAG:
    """
    Tool-calling agent that decides when to use RAG.

    Tools:
      1. retrieve_documents(query): fetch relevant indexed chunks
      2. get_collection_info(): inspect indexed collection metadata
    """

    def __init__(self, retriever: Retriever, vector_store: VectorStore) -> None:
        self.retriever = retriever
        self.vector_store = vector_store
        self._last_documents: list[Document] = []
        self.agent = self._build_agent()

    def invoke(self, query: str) -> dict:
        """Run the agent and return the final answer plus any retrieved docs."""
        self._last_documents = []
        result = self.agent.invoke({"messages": [{"role": "user", "content": query}]})
        answer = self._extract_final_text(result["messages"][-1].content)
        route = "document" if self._last_documents else "general"
        return {
            "query": query,
            "answer": answer,
            "documents": self._last_documents,
            "query_type": route,
            "top_score": (
                self._last_documents[0].metadata.get("rerank_score")
                if self._last_documents
                else None
            ),
        }

    def _build_agent(self):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "Missing GROQ_API_KEY. Add it to a .env file or set it in your "
                "terminal before running the ask command."
            )

        model = ChatOpenAI(
            model=config.GROQ_MODEL_NAME,
            api_key=config.GROQ_API_KEY,
            base_url=config.GROQ_BASE_URL,
            temperature=config.LLM_TEMPERATURE,
        )

        @tool
        def retrieve_documents(query: str) -> str:
            """Search the indexed documents and return the most relevant chunks."""
            documents = self.retriever.retrieve(query)
            self._last_documents = documents
            if not documents:
                return "No relevant indexed documents were found."

            sections = []
            for i, doc in enumerate(documents, 1):
                source = doc.metadata.get("source", "unknown")
                page = doc.metadata.get("page", "?")
                score = doc.metadata.get("rerank_score", "N/A")
                sections.append(
                    f"[{i}] source={source} page={page} score={score}\n"
                    f"{doc.page_content[:700]}"
                )
            return "\n\n".join(sections)

        @tool
        def get_collection_info() -> str:
            """Return metadata about the indexed document collection."""
            info = self.vector_store.info()
            return (
                f"collection={info['collection']}\n"
                f"persist_dir={info['persist_dir']}\n"
                f"total_documents={info['total_documents']}\n"
                f"supported_extensions={config.SUPPORTED_EXTENSIONS}"
            )

        system_prompt = (
            "You are an agentic RAG assistant.\n"
            "Use the retrieve_documents tool when the user asks about uploaded or "
            "indexed documents, data, reports, CSVs, PDFs, or anything that may be "
            "inside the local knowledge base.\n"
            "Use get_collection_info when the user asks what is indexed or available.\n"
            "If the user asks a general question that does not depend on the uploaded "
            "documents, answer directly without calling tools.\n"
            "When you use retrieve_documents, ground your answer in the returned "
            "chunks and cite source/page where possible."
        )

        return create_agent(
            model=model,
            tools=[retrieve_documents, get_collection_info],
            system_prompt=system_prompt,
        )

    @staticmethod
    def _extract_final_text(content) -> str:
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif hasattr(block, "get") and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                else:
                    text_parts.append(str(block))
            return "\n".join(part for part in text_parts if part).strip()

        return str(content).strip()
