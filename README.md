# RAG Pipeline — ML Team Task

> **Stack:** LangChain · Marker (PDF) · ChromaDB · SentenceTransformers · CrossEncoder

---

##  Project Structure

```
rag_project/
├── main.py                        ← CLI entry point
├── config.py                      ← All settings & paths
├── pyproject.toml                 ← Project metadata & uv dependencies
├── uv.lock                        ← Locked dependency versions
├── README.md
├── data/                          ← Drop your files here
├── chroma_db/                     ← Persistent vector store (auto-created)
├── logs/                          ← Daily log files  (auto-created)
├── qa_history/                    ← Q&A JSON history (auto-created)
└── src/
    ├── loader/document_loader.py  ← PDF (Marker) / CSV / TXT / DOCX
    ├── logger/log_setup.py        ← LoggerFactory
    ├── chunker/text_chunker.py    ← RecursiveCharacterTextSplitter
    ├── embedder/embedder.py       ← HuggingFace sentence-transformers
    ├── vectorstore/chroma_store.py← ChromaDB (persistent)
    ├── reranker/reranker.py       ← CrossEncoder re-ranker
    ├── retriever/retriever.py     ← Vector search → re-rank pipeline
    ├── graph/rag_graph.py         ← LangGraph RAG workflow
    └── qa/qa_engine.py            ← Ask questions, save history
```

---

##  Setup

```bash
# Install dependencies with uv
uv sync
```

Create a local `.env` file for Groq:

```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL_NAME=llama-3.3-70b-versatile
LLM_TEMPERATURE=0.2
DOC_RELEVANCE_THRESHOLD=-11.0
```

The agentic RAG workflow routes each question before generation:

```
agent → decide whether to call a tool → document answer OR general Groq answer
```

Defined tools:

1. `retrieve_documents(query)` → fetches relevant indexed chunks
2. `get_collection_info()` → returns indexed collection metadata

If the question is about uploaded documents, the agent calls `retrieve_documents`.
If the question is about what is indexed, it can call `get_collection_info`.
If the question is general, it answers directly without using tools.

---

##  Usage

### Index files
```bash
python main.py index --files data/report.pdf data/data.csv data/notes.txt
```

### Ask a question
```bash
python main.py ask --query "What is the main finding?"
```

### Ask without re-ranking
```bash
python main.py ask --query "Summarise the key points" --no-rerank
```

### View Q&A history
```bash
python main.py history
python main.py history --last-n 5
```

### Vector store info
```bash
python main.py info
```

### Clear the vector store
```bash
python main.py clear
```

---

##  Use as a Python API

```python
from src.loader.document_loader import DocumentLoader
from src.chunker.text_chunker import TextChunker
from src.vectorstore.chroma_store import VectorStore
from src.reranker.reranker import Reranker
from src.retriever.retriever import Retriever
from src.qa.qa_engine import QAEngine

# Build pipeline
vector_store = VectorStore()
retriever    = Retriever(vector_store, Reranker())
engine       = QAEngine(retriever)

# Index
loader  = DocumentLoader()
chunker = TextChunker()
docs    = loader.load("data/report.pdf")
chunks  = chunker.split(docs)
vector_store.add_documents(chunks)

# Ask
engine.ask("What is this report about?")
```

---

##  Config Tuning (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `INITIAL_K` | 10 | Candidates from ChromaDB |
| `FINAL_K` | 3 | Results after re-ranking |
| `EMBED_DEVICE` | `cpu` | Set to `cuda` for GPU |

---

##  Supported File Types

| Extension | Parser |
|---|---|
| `.pdf` | Marker (markdown-quality extraction) |
| `.csv` | LangChain CSVLoader |
| `.txt` | LangChain TextLoader |
| `.docx` | LangChain Docx2txtLoader |
