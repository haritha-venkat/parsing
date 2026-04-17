# Mining Capex RAG Pipeline

> **Stack:** LangChain · PyMuPDF · ChromaDB · SentenceTransformers · CrossEncoder

A Retrieval-Augmented Generation (RAG) pipeline for extracting, indexing, and querying capital expenditure (capex) data from mining company annual reports. Includes specialized scripts for table extraction and structured summaries.

---

## Project Structure

```
rag_project/
├── main.py                        ← CLI entry point for RAG Q&A
├── capex_table_extractor.py       ← Extract capex tables or text from PDFs
├── capex_summary.py               ← Generate structured capex summaries
├── config.py                      ← All settings & paths
├── pyproject.toml                 ← Project dependencies & scripts
├── README.md
├── data/                          ← PDF files (e.g., annual reports)
├── chroma_db/                     ← Persistent vector store (auto-created)
├── logs/                          ← Daily log files (auto-created)
├── capex_tables.xlsx              ← Extracted tables (auto-generated)
├── {Company}_capex.txt            ← Fallback text for PDFs without tables
└── src/
    ├── __init__.py
    ├── logger/log_setup.py        ← LoggerFactory
    ├── loader/capex_loader.py      ← PDF loader for capex documents
    ├── chunker/text_chunker.py    ← RecursiveCharacterTextSplitter
    ├── embedder/embedder.py       ← HuggingFace sentence-transformers
    ├── vectorstore/chroma_store.py← ChromaDB (persistent)
    ├── reranker/reranker.py       ← CrossEncoder re-ranker
    ├── retriever/retriever.py     ← Vector search → re-rank pipeline
    ├── qa/qa_engine.py            ← Ask questions, save history
    ├── agentic/
    │   ├── __init__.py
    │   └── agentic_rag.py         ← Advanced agentic RAG
    └── graph/
        ├── __init__.py
        └── rag_graph.py           ← LangGraph-based RAG workflow
```

### Extract Capex Data from PDFs
```bash
python capex_table_extractor.py
```
- Extracts structured tables to `capex_tables.xlsx` for companies with tables.
- For PDFs without tables, extracts relevant $ amount lines to `{Company}_capex.txt`.

### Generate Capex Summaries
```bash
python capex_summary.py
```
- Creates structured Excel summaries with total, sustaining, growth, and development capex figures.

### Index Documents for RAG
```bash
python main.py index --files data/antofagasta-2022-ara.pdf data/Barrick_Annual_Report_2022.pdf
```

### Ask a Question
```bash
python main.py ask --query "What was Antofagasta's total capital expenditure in 2022?"
```

### Ask without Re-ranking
```bash
python main.py ask --query "Summarize capex trends" --no-rerank
```

### View Q&A History
```bash
python main.py history
python main.py history --last-n 5
```

### Vector Store Info
```bash
python main.py info
```

### Clear the Vector Store
```bash
python main.py clear
```

---

## Use as a Python API

```python
from src.loader.capex_loader import load_capex_batch
from src.chunker.text_chunker import TextChunker
from src.vectorstore.chroma_store import VectorStore
from src.reranker.reranker import Reranker
from src.retriever.retriever import Retriever
from src.qa.qa_engine import QAEngine

# Build pipeline
vector_store = VectorStore()
retriever    = Retriever(vector_store, Reranker())
engine       = QAEngine(retriever)

# Index capex PDFs
docs    = load_capex_batch("data/")
chunker = TextChunker()
chunks  = chunker.split(docs)
vector_store.add_documents(chunks)

# Ask about capex
engine.ask("What are the capex figures for Barrick Gold?")
```

---

## Config Tuning (`config.py`)

| Parameter | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 500 | Characters per chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `INITIAL_K` | 10 | Candidates from ChromaDB |
| `FINAL_K` | 3 | Results after re-ranking |
| `EMBED_DEVICE` | `cpu` | Set to `cuda` for GPU |

---

## Dependencies

Install with:
```bash
pip install -e .
```
Or using uv:
```bash
uv sync
```

Key packages: `langchain`, `chromadb`, `pymupdf`, `pandas`, `openpyxl`, `sentence-transformers`.

---

##  Supported File Types

| Extension | Parser |
|---|---|
| `.pdf` | Marker (markdown-quality extraction) |
| `.csv` | LangChain CSVLoader |
| `.txt` | LangChain TextLoader |
| `.docx` | LangChain Docx2txtLoader |
