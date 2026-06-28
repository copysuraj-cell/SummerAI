# Observability Comparison: Logfire vs Arize

This app runs the same FAISS RAG pipeline side by side, sending traces to two different observability platforms — **Logfire** on the left and **Arize** on the right. The goal is to compare how each platform surfaces what's happening inside a RAG system.

---

## What it does

Upload a PDF or text file, ask a question, and both columns answer using the same FAISS index and the same Grok LLM. The only difference is where the trace data goes.

```
Upload document → chunk text → build FAISS index
                                      ↓
User asks question → retrieve top-3 chunks → augment prompt → generate answer (grok-3-mini)
         ↓                                                              ↓
   Logfire span                                                   Arize span
   (left column)                                                 (right column)
```

---

## Logfire vs Arize — How They Differ

Both platforms receive the same span data, but they surface it differently.

### Logfire
- Built by Pydantic, designed around structured logging with OpenTelemetry under the hood
- Uses `logfire.span()` and `logfire.info()` — feels like structured logging with trace context attached
- Spans show up in a live timeline view; attributes like `query` and `answer` appear on the root span
- Dashboard at [logfire.pydantic.dev](https://logfire.pydantic.dev)

### Arize
- Built for LLM/AI observability specifically — understands concepts like retrievers, LLM calls, and chains
- Uses raw OpenTelemetry spans with **OpenInference** semantic attributes (`openinference.span.kind`, `input.value`, `output.value`, `retrieval.documents.*`)
- These typed attributes let Arize render Input/Output columns, retrieved documents, and LLM model info automatically
- Dashboard at [app.arize.com](https://app.arize.com) — traces land under the `faiss-rag-observability` project

### Instrumentation comparison

| | Logfire | Arize |
|---|---|---|
| API style | `logfire.span()`, `logfire.info()` | Raw OTEL `tracer.start_as_current_span()` |
| Semantic standard | Logfire's own attribute names | OpenInference spec |
| Span kinds | Free-form names | Typed: `CHAIN`, `RETRIEVER`, `LLM` |
| Input/Output | Set as span attributes (`query`, `answer`) | `input.value` / `output.value` attributes |
| Retrieved docs | Logged as `chunk_count` info event | `retrieval.documents.{i}.document.content` |
| Dashboard | Cloud (logfire.pydantic.dev) | Cloud (app.arize.com) |
| Flush behavior | Requires `logfire.force_flush()` in long-running apps | `batch=False` flushes immediately per span |

---

## Setup

### 1. Clone the repo and navigate into it

```bash
git clone <repo-url>
cd SummerAI
```

### 2. Create and activate the virtual environment

```bash
cd observability
python3 -m venv obsenv
source obsenv/bin/activate      # Mac/Linux
# obsenv\Scripts\activate       # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your API keys

You need keys from three services:

**xAI (for Grok generation)**
1. Go to [x.ai](https://x.ai) and sign in
2. Navigate to API → Create API Key
3. Copy the key

**Logfire**
1. Go to [logfire.pydantic.dev](https://logfire.pydantic.dev) and sign up
2. Create a new project
3. Go to Settings → API Keys → Create token
4. Copy the token

**Arize**
1. Go to [app.arize.com](https://app.arize.com) and sign up (free)
2. Go to Settings → API Keys
3. Copy your **Space ID** and **API Key** separately — you need both

### 5. Create a `.env` file

In the `SummerAI/` root folder, create a file called `.env` and paste in your keys:

```
XAI_API_KEY=your_xai_key_here
LOGFIRE_TOKEN=your_logfire_token_here
ARIZE_SPACE_ID=your_arize_space_id_here
ARIZE_API_KEY=your_arize_api_key_here
```

### 6. Run the app

Make sure your virtual environment is still active, then from inside the `observability/` folder:

```bash
source obsenv/bin/activate
streamlit run observability_comparison.py
```

The app will open automatically at `http://localhost:8501`. If it doesn't, open that URL in your browser manually.

### 7. Use it

1. Upload a PDF or `.txt` file using the file uploader
2. Wait a few seconds for the FAISS index to build (only happens once per file)
3. Type a question in the chat bar at the bottom
4. Both columns will answer — Logfire traces the left, Arize traces the right
5. Open each dashboard to see the traces:
   - Logfire: [logfire.pydantic.dev](https://logfire.pydantic.dev) → your project → Live
   - Arize: [app.arize.com](https://app.arize.com) → Tracing → `faiss-rag-observability`

---

## Viewing traces

**Logfire:** Go to [logfire.pydantic.dev](https://logfire.pydantic.dev) → your project → Live view. Click any trace to expand spans. The root `faiss.rag` span has `query` and `answer` as attributes.

**Arize:** Go to [app.arize.com](https://app.arize.com) → Tracing → `faiss-rag-observability` project. The trace list shows Input and Output columns populated directly from the `input.value` / `output.value` attributes. Click into a trace to see the `RETRIEVER` span with the retrieved document chunks.

---

## Architecture

### FAISS RAG pipeline

Documents are split on double newlines (`\n\n`). For PDFs, extraction is page-by-page via `pypdf`. Chunks shorter than 80 characters are dropped. Each chunk is embedded with `all-MiniLM-L6-v2` (384-dim, local) into a `faiss.IndexFlatIP`. Retrieval is exact cosine similarity (vectors are L2-normalised). The top 3 chunks are used as context.

### Observability wiring

Both columns share a single cached FAISS index (`@st.cache_resource`). The observability setup is also cached so Logfire and Arize are only registered once per server lifetime — not on every Streamlit re-run.

- **Logfire** sets the global OTEL tracer provider via `logfire.configure()`
- **Arize** uses a separate `TracerProvider` with `set_global_tracer_provider=False` to avoid conflicting with Logfire

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `faiss-cpu` | Vector similarity search |
| `sentence-transformers` | Local embeddings (`all-MiniLM-L6-v2`) |
| `openai` | xAI Grok API client (OpenAI-compatible) |
| `pypdf` | PDF text extraction |
| `logfire` | Logfire tracing |
| `arize-otel` | Arize OpenTelemetry registration |
| `python-dotenv` | Load `.env` file |
