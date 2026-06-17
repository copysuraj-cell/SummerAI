# SummerAI — FAISS vs Cognee RAG Comparison

This project compares two different approaches to **Retrieval-Augmented Generation (RAG)** side by side: a FAISS vector database and a Cognee knowledge graph. The goal is to understand how each system decides what information to pull from a document — and how that difference affects the answers you get.

---

## What is RAG?

RAG stands for **Retrieval-Augmented Generation**. Instead of asking an AI to answer from memory, RAG first searches a knowledge base for relevant passages, then gives those passages to the model as context. The model's answer is only as good as what was retrieved.

```
Upload document → split into chunks → index chunks
                                            ↓
User asks question → find relevant chunks → add to prompt → generate answer
```

The retrieval step is where FAISS and Cognee fundamentally differ.

---

## FAISS vs Cognee — How They Differ

Both systems read the same document and answer the same question. But they retrieve information in completely different ways.

### FAISS — Vector Similarity Search

FAISS converts every chunk of text into a list of 384 numbers (a vector) that captures its meaning. When you ask a question, it converts your question into the same kind of vector and finds the chunks whose vectors are mathematically closest.

- **What it retrieves:** the top-3 chunks with the highest cosine similarity score to your question
- **Strength:** fast and precise — great at finding passages that directly match your question's wording or topic
- **Weakness:** treats each chunk independently; doesn't understand how concepts relate to each other across the document
- **Embedding model:** `all-MiniLM-L6-v2` (runs locally)
- **Score:** each retrieved chunk has a similarity score between 0 and 1 (shown in the UI)

### Cognee — Knowledge Graph

Cognee reads the document and builds a **graph** of concepts and the relationships between them, powered by Grok. When you ask a question, it traverses that graph to find relevant nodes rather than scanning all chunks.

- **What it retrieves:** nodes from the knowledge graph — could be concepts, entities, or relationships, not just raw text chunks
- **Strength:** better at questions that require connecting ideas across different parts of the document
- **Weakness:** slower to build, and graph quality depends on how well Grok understood the document
- **Embedding model:** `nomic-embed-text` (via Ollama, 768-dim)
- **Internal LLM:** Grok `grok-3-mini` (via xAI API) — used to build the graph, not to answer

### At a glance

| | FAISS | Cognee |
|---|---|---|
| Index type | Flat vector index | Knowledge graph |
| Retrieval method | Nearest-neighbor cosine similarity | Graph traversal |
| Retrieved output | Raw text chunks with similarity scores | Graph nodes (concepts/relationships) |
| Embedding model | `all-MiniLM-L6-v2` — local, 384-dim | `nomic-embed-text` — local, 768-dim |
| Needs external API | No | Yes (xAI/Grok for graph building) |
| Index build time | Fast (seconds) | Slow (graph construction) |
| Best for | Direct factual lookup | Relational or multi-concept questions |

---

## The Comparison App

**`rag/rag_comparison.py`** is the main app. It runs both pipelines on the same question and shows:

- The answer from each system
- An expandable **"What was retrieved"** section under each answer, showing exactly which chunks or graph nodes each system pulled — this is where you can see *why* the answers differ
- The similarity score for each FAISS chunk (so you can see how confident it was)
- Response time for each system

```bash
streamlit run rag/rag_comparison.py
```

The other two apps run each system independently:

```bash
streamlit run rag/rag_with_vector_db.py   # FAISS only
streamlit run rag/rag_with_cognee.py      # Cognee only
```

---

## Setup

### 1. Create and activate the virtual environment
```bash
python3 -m venv ragenv
source ragenv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Pull required Ollama models
```bash
ollama pull minimax-m3:cloud      # generation model (all apps)
ollama pull nomic-embed-text      # embedding model (Cognee apps)
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
XAI_API_KEY=your_xai_api_key_here
```
Get a key at [x.ai](https://x.ai). It's used by Cognee internally to build the knowledge graph via Grok. The generation model (`minimax-m3:cloud`) runs locally through Ollama and doesn't need an API key.

> `.env` is gitignored — never commit it.

---

## Usage

1. Run the comparison app with `streamlit run rag/rag_comparison.py`
2. Upload a `.pdf` or `.txt` file using the file uploader
3. Wait for both indexes to build (spinners show progress)
4. Ask a question — both systems answer simultaneously
5. Expand **"What FAISS retrieved"** and **"What Cognee retrieved"** to see how each system approached the question

Uploading a new file automatically rebuilds both indexes and clears the chat history.

---

## Project Structure

```
SummerAI/
├── rag/
│   ├── rag_comparison.py       # Main app: side-by-side FAISS vs Cognee
│   ├── rag_with_vector_db.py   # FAISS-only app
│   └── rag_with_cognee.py      # Cognee-only app
├── ragenv/                     # Python virtual environment
├── requirements.txt            # Direct dependencies
├── .env                        # API keys (gitignored)
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Local text embeddings for FAISS |
| `cognee` | Knowledge graph RAG backend |
| `ollama` | Local LLM inference |
| `pypdf` | PDF text extraction |
| `numpy` | Vector math |
| `python-dotenv` | Load `.env` file |
