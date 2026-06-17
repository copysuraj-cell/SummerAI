#!/usr/bin/env python3
"""
Side-by-side comparison of FAISS RAG vs Cognee RAG.

Run: streamlit run rag_comparison.py
"""

import os
import asyncio
import tempfile
import textwrap
from dotenv import load_dotenv

load_dotenv()

# Cognee env vars — must be set before importing cognee
os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
os.environ["LLM_PROVIDER"]               = "openai"
os.environ["LLM_MODEL"]                  = "openai/grok-3-mini"
os.environ["LLM_ENDPOINT"]              = "https://api.x.ai/v1"
os.environ["LLM_API_KEY"]               = os.getenv("XAI_API_KEY", "")
os.environ["EMBEDDING_PROVIDER"]         = "litellm"
os.environ["EMBEDDING_MODEL"]           = "ollama/nomic-embed-text"
os.environ["EMBEDDING_DIMENSIONS"]       = "768"
os.environ["HUGGINGFACE_TOKENIZER"]      = "bert-base-uncased"

from pypdf import PdfReader
import streamlit as st
import numpy as np
import faiss
import ollama
import cognee
from sentence_transformers import SentenceTransformer

CHAT_MODEL   = "minimax-m3:cloud"
PROFILE_FILE = "suraj_profile_summary.txt"

# ── Shared helpers ────────────────────────────────────────────────────────────

def load_docs(path: str) -> list[dict]:
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        chunks = []
        for page in reader.pages:
            text = page.extract_text() or ""
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 80:
                    chunks.append(para)
    else:
        text = open(path).read()
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [{"id": str(i + 1), "source": os.path.basename(path), "text": chunk}
            for i, chunk in enumerate(chunks)]

def load_profile(path: str) -> str:
    try:
        return open(path).read().strip()
    except FileNotFoundError:
        return ""

def augment(query: str, chunks: list[str]) -> str:
    context = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(chunks))
    return textwrap.dedent(f"""\
        Answer using ONLY the context below. Cite [1], [2], … per fact.
        Say "I don't have that information" if the answer isn't there.

        {context}

        Question: {query}""")

def generate(prompt: str) -> str:
    profile = load_profile(PROFILE_FILE)
    messages = []
    if profile:
        messages.append({"role": "system", "content": f"User profile:\n{profile}"})
    messages.append({"role": "user", "content": prompt})
    return ollama.chat(model=CHAT_MODEL, messages=messages)["message"]["content"]

# ── FAISS pipeline ────────────────────────────────────────────────────────────

class VectorStore:
    def build(self, docs: list[dict], model: SentenceTransformer) -> None:
        self._docs = docs
        self._model = model
        vecs = self._embed([d["text"] for d in docs])
        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        q = self._embed([query])
        scores, indices = self._index.search(q, top_k)
        return [self._docs[i]["text"] for i in indices[0] if i != -1]

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vecs)
        return vecs

@st.cache_resource
def faiss_load_store(_tmp_path: str, file_key: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = load_docs(_tmp_path)
    store = VectorStore()
    store.build(docs, model)
    return store

def faiss_rag(query: str) -> str:
    chunks = faiss_load_store(st.session_state["tmp_path"], st.session_state["file_key"]).search(query)
    return generate(augment(query, chunks))

# ── Cognee pipeline ───────────────────────────────────────────────────────────

@st.cache_resource
def cognee_build_index(_tmp_path: str, file_key: str):
    async def _build():
        await cognee.forget(everything=True)
        for doc in load_docs(_tmp_path):
            await cognee.remember(doc["text"])
    asyncio.run(_build())

def cognee_rag(query: str) -> str:
    results = asyncio.run(cognee.recall(query))
    chunks = [r.text if hasattr(r, "text") else str(r) for r in results[:3]]
    return generate(augment(query, chunks))

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide")
st.title("RAG Comparison: FAISS vs Cognee")

uploaded = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

if uploaded is None:
    st.info("Upload a PDF or text file to start comparing.")
    st.stop()

file_key = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("file_key") != file_key:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        st.session_state["tmp_path"] = f.name
    st.session_state["file_key"] = file_key
    st.session_state["messages"] = []

col_faiss, col_cognee = st.columns(2)
col_faiss.subheader("FAISS")
col_cognee.subheader("Cognee")

with st.spinner("Loading FAISS index..."):
    faiss_load_store(st.session_state["tmp_path"], file_key)

with st.spinner("Building Cognee knowledge graph..."):
    cognee_build_index(st.session_state["tmp_path"], file_key)

st.caption(f"Knowledge base: `{uploaded.name}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    col = col_faiss if msg["col"] == "faiss" else col_cognee
    with col.chat_message(msg["role"]):
        col.write(msg["content"])

if query := st.chat_input("Ask a question..."):
    col_faiss.chat_message("user").write(query)
    col_cognee.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query, "col": "faiss"})
    st.session_state.messages.append({"role": "user", "content": query, "col": "cognee"})

    with col_faiss.chat_message("assistant"):
        with st.spinner("FAISS thinking..."):
            faiss_answer = faiss_rag(query)
        col_faiss.write(faiss_answer)

    with col_cognee.chat_message("assistant"):
        with st.spinner("Cognee thinking..."):
            cognee_answer = cognee_rag(query)
        col_cognee.write(cognee_answer)

    st.session_state.messages.append({"role": "assistant", "content": faiss_answer,  "col": "faiss"})
    st.session_state.messages.append({"role": "assistant", "content": cognee_answer, "col": "cognee"})
