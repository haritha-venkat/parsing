# RAG Pipeline — ML Team Task

> **Stack:** LangChain · Marker (PDF) · ChromaDB · SentenceTransformers · CrossEncoder

---

##  Project Structure

```
rag_project/
├── main.py                        ← CLI entry point
├── config.py                      ← All settings & paths
├── requirements.txt
├── README.md
├── data/                          ← Drop your files here
├── chroma_db/                     ← Persistent vector store (auto-created)
├── logs/                          ← Daily log files  (auto-created)
├── qa_history/                    ← Q&A JSON history (auto-created)
└── src/
    ├── logger/log_setup.py        ← LoggerFactory
    ├── loader/document_loader.py  ← PDF (Marker) / CSV / TXT / DOCX
    ├── chunker/text_chunker.py    ← RecursiveCharacterTextSplitter
    ├── embedder/embedder.py       ← HuggingFace sentence-transformers
    ├── vectorstore/chroma_store.py← ChromaDB (persistent)
    ├── reranker/reranker.py       ← CrossEncoder re-ranker
    ├── retriever/retriever.py     ← Vector search → re-rank pipeline
    └── qa/qa_engine.py            ← Ask questions, save history
```

---

##  Setup

```bash
# 1. Create virtual environment and install dependencies with uv
uv sync

# Optional fallback if uv is not available
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

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
