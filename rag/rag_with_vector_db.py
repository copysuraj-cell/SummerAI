#!/usr/bin/env python3
"""
RAG (Retrieval-Augmented Generation) with a vector database.

Pipeline: documents → embed → vector DB → query → retrieve → augment prompt → generate

Install: pip install ollama faiss-cpu sentence-transformers streamlit
Run:     streamlit run rag_with_vector_db.py
"""

import os
import tempfile
import textwrap

from pypdf import PdfReader
import streamlit as st
import numpy as np
import faiss
import ollama
from sentence_transformers import SentenceTransformer

CHAT_MODEL   = "minimax-m3:cloud"
PROFILE_FILE = "suraj_profile_summary.txt"

def load_profile(path: str) -> str:
    try:
        return open(path).read().strip()
    except FileNotFoundError:
        return ""

# ── Knowledge base ────────────────────────────────────────────────────────────

def load_docs(path: str) -> list[dict]:
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        chunks = []
        for page in reader.pages:
            text = page.extract_text() or ""
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 80:  # skip headers/page numbers
                    chunks.append(para)
    else:
        text = open(path).read()
        chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [{"id": str(i + 1), "source": os.path.basename(path), "text": chunk}
            for i, chunk in enumerate(chunks)]

# ── Vector store (FAISS) ──────────────────────────────────────────────────────
# sentence-transformers produces dense 384-dim vectors.
# IndexFlatIP does exact inner-product search; on L2-normalised vectors this
# equals cosine similarity.

class VectorStore:
    def build(self, docs: list[dict], model: SentenceTransformer) -> None:
        self._docs = docs
        self._model = model
        vecs = self._embed([d["text"] for d in docs])
        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = self._embed([query])
        scores, indices = self._index.search(q, top_k)
        return [
            {**self._docs[i], "score": float(scores[0][rank])}
            for rank, i in enumerate(indices[0])
            if i != -1
        ]

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vecs)
        return vecs

# ── RAG pipeline ──────────────────────────────────────────────────────────────

def retrieve(store: VectorStore, query: str, top_k: int = 3) -> list[dict]:
    return store.search(query, top_k)



def augment(query: str, docs: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] ({d['source']})\n{d['text']}" for i, d in enumerate(docs)
    )
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
    response = ollama.chat(model=CHAT_MODEL, messages=messages)
    return response["message"]["content"]

def rag(store: VectorStore, query: str) -> dict:
    docs = retrieve(store, query)
    prompt = augment(query, docs)
    answer = generate(prompt)
    return {"query": query, "retrieved": docs, "answer": answer}

# ── Streamlit UI ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_store(_tmp_path: str, file_key: str):
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = load_docs(_tmp_path)
    store = VectorStore()
    store.build(docs, model)
    return store, len(docs)

st.title("RAG Chatbot")

uploaded = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])

if uploaded is None:
    st.info("Upload a PDF or text file to start chatting.")
    st.stop()

file_key = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("file_key") != file_key:
    suffix = os.path.splitext(uploaded.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(uploaded.read())
        st.session_state["tmp_path"] = f.name
    st.session_state["file_key"] = file_key
    st.session_state["messages"] = []

with st.spinner("Loading model and knowledge base..."):
    store, n_docs = load_store(st.session_state["tmp_path"], file_key)

st.caption(f"{n_docs} chunks loaded from `{uploaded.name}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if query := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = rag(store, query)
        st.write(result["answer"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
