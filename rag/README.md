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

It uses `qwen2.5:0.5b` via a remote Ollama server at `http://192.168.0.34:11434` for generation, and is fully instrumented with **Logfire** — every index build, retrieval call, and LLM generation is traced as a named span.

```bash
streamlit run rag/rag_comparison.py
```

The other two apps run each system independently:

```bash
streamlit run rag/rag_with_vector_db.py   # FAISS only
streamlit run rag/rag_with_cognee.py      # Cognee only
```

**`rag/observability_comparison.py`** is a second comparison that keeps Cognee as the RAG backend but swaps the observability layer:

- **Left column — Logfire:** uses `logfire.span()` / `logfire.info()` and sends traces to the Logfire cloud dashboard
- **Right column — Arize Phoenix:** uses raw OpenTelemetry spans sent to a local Phoenix server that launches automatically on startup

```bash
streamlit run rag/observability_comparison.py
```

After launch, dashboard links for both platforms appear above the chat columns.

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
ollama pull qwen2.5:0.5b          # generation model used by rag_comparison.py
ollama pull nomic-embed-text      # embedding model (Cognee apps)
```

### 4. Set up environment variables
Create a `.env` file in the project root:
```
XAI_API_KEY=your_xai_api_key_here
LOGFIRE_TOKEN=your_logfire_token_here   # optional — only needed for Logfire dashboard
```
Get an xAI key at [x.ai](https://x.ai). It's used by Cognee internally to build the knowledge graph via Grok.

> `.env` is gitignored — never commit it.

### 5. Authenticate Logfire (optional)
`rag_comparison.py` is instrumented with [Logfire](https://logfire.pydantic.dev) for tracing. Without auth, spans print to the terminal. To send traces to the dashboard:
```bash
source ragenv/bin/activate
logfire auth
```
This opens a browser sign-in and saves a token locally. No code changes needed after that.

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
│   ├── rag_comparison.py           # Side-by-side FAISS vs Cognee
│   ├── observability_comparison.py # Logfire vs Arize Phoenix (both use Cognee)
│   ├── rag_with_vector_db.py       # FAISS-only app
│   └── rag_with_cognee.py          # Cognee-only app
├── family-finance-os/          # ← NEW: Premium family expense analytics dashboard (vanilla JS SPA)
│   ├── index.html              # Single HTML file — all views as hidden <section>s
│   ├── css/vars.css            # Design tokens (dark slate/indigo theme)
│   ├── css/landing.css         # Landing + login styles (GSAP animations, 3D tilt, cursor glow)
│   ├── css/app.css             # App shell: sidebar, dashboard, expenses table, modals
│   ├── js/config.js            # Firebase config (fill in from Firebase Console)
│   ├── js/data.js              # 424 mock expense records + LIFE_AREA constants
│   ├── js/analytics.js         # Pure analytics functions (totals, trends, anomalies, YoY)
│   ├── js/auth.js              # Firebase Auth (Google + email/password) + Firestore instance
│   ├── js/charts.js            # SVG chart builders (area, bar, donut, grouped bar, sparkline)
│   ├── js/query.js             # NLP intent engine — answers plain-English spending questions
│   └── js/app.js               # SPA controller: routing, all view renderers, event wiring
├── screenshot demo/            # Static explainer website
│   ├── index.html              # Page structure (6 sections)
│   ├── style.css               # Design tokens, dark/light themes
│   └── main.js                 # Scroll-synced animations, particle canvas, demo
├── snack-tracker/              # Vanilla JS snack tracking app (Firebase)
├── ragenv/                     # Python virtual environment
├── requirements.txt            # Direct dependencies
├── .env                        # API keys (gitignored)
└── README.md
```

### Family Finance OS (`family-finance-os/`)

A premium family expense analytics dashboard — vanilla JS SPA, no build step, no npm. Backed by **Firebase Auth + Firestore** so anyone can sign up and their data lives in the cloud.

**Setup** — paste your Firebase config into `js/config.js` (see Firebase setup steps below).

Features:
- **Breathtaking landing page** — GSAP clip-path word reveal, 3D mouse-tilt card, magnetic buttons, cursor glow with lerp smoothing, particle canvas
- **Cloud auth** — Google Sign-In + email/password registration via Firebase Auth; each user's data is isolated under `users/{uid}/expenses` in Firestore
- **Smart mock data** — 424 pre-loaded sample records show only when a new user has zero real expenses; first real write permanently exits mock mode
- **Dashboard** — 4 stat cards, 12-month area chart, life-area donut, biggest purchases, YoY grouped bar chart
- **Expenses** — search/filter/sort table with pagination, CSV export, add/edit/delete modal (all writes go to Firestore)
- **Categories** — 10 life area cards → drilldown with merchant bars and monthly chart
- **Insights** — auto-insights, time machine period comparison, multi-year trends
- **Ask AI** — plain-English spending questions answered from your data (no API key needed)
- **Import** — 4-step CSV wizard with auto column-mapping (batch writes to Firestore)

#### Firebase setup (one-time)

1. Go to [console.firebase.google.com](https://console.firebase.google.com) → **Create project**
2. **Authentication** → Sign-in methods → enable **Google** and **Email/Password**
3. **Firestore Database** → Create database → Start in **production mode**
4. Paste these security rules and publish:
   ```
   rules_version = '2';
   service cloud.firestore {
     match /databases/{database}/documents {
       match /users/{userId}/{document=**} {
         allow read, write: if request.auth != null && request.auth.uid == userId;
       }
     }
   }
   ```
5. **Project Settings** → Your apps → Add web app → copy the config object
6. Paste it into `js/config.js`:
   ```js
   const CONFIG = {
     firebase: {
       apiKey: '...',
       authDomain: '...',
       projectId: '...',
       storageBucket: '...',
       messagingSenderId: '...',
       appId: '...',
     }
   };
   ```
7. Deploy to Netlify (drag the `family-finance-os/` folder) → add the Netlify domain to **Firebase Auth → Authorized domains**

---

### Website (`screenshot demo/`)

A static single-page site explaining the FAISS vs Cognee comparison. Open `index.html` directly in a browser — no build step needed.

Features:
- Particle canvas with mouse interaction in the hero
- **Scroll-synced animations throughout**: every section animates its elements as you scroll into them — hero parallax, pipeline steps staggering in, compare cards sliding from opposite sides, stack cards cascading
- Gradient scroll-progress bar at the top of the page
- Active nav link highlighting based on scroll position
- Interactive Q&A demo with 4 preset questions showing side-by-side FAISS vs Cognee responses
- Dark/light theme toggle (persists across reloads)

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
| `logfire` | Tracing and observability for `rag_comparison.py` |
| `arize-phoenix` | Local observability server for `observability_comparison.py` |
| `arize-phoenix-otel` | Phoenix OTEL helpers |
| `opentelemetry-exporter-otlp-proto-http` | OTLP HTTP exporter (Phoenix backend) |
