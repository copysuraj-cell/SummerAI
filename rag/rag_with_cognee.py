#!/usr/bin/env python3
"""
RAG with Cognee + Ollama.

Run: streamlit run rag_with_cognee.py
Requires: ollama pull nomic-embed-text
"""

import os
import asyncio
import tempfile
import textwrap
from dotenv import load_dotenv

load_dotenv()  # must run before os.getenv() calls below

os.environ["COGNEE_SKIP_CONNECTION_TEST"] = "true"
os.environ["LLM_PROVIDER"] = "openai"
os.environ["LLM_MODEL"] = "openai/grok-3-mini"
os.environ["LLM_ENDPOINT"] = "https://api.x.ai/v1"
os.environ["LLM_API_KEY"] = os.getenv("XAI_API_KEY", "")
os.environ["EMBEDDING_PROVIDER"] = "litellm"
os.environ["EMBEDDING_MODEL"] = "ollama/nomic-embed-text"
os.environ["EMBEDDING_DIMENSIONS"] = "768"
os.environ["HUGGINGFACE_TOKENIZER"] = "bert-base-uncased"

from pypdf import PdfReader
import streamlit as st
import ollama
import cognee

CHAT_MODEL   = "minimax-m3:cloud"
PROFILE_FILE = "suraj_profile_summary.txt"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_profile(path: str) -> str:
    try:
        return open(path).read().strip()
    except FileNotFoundError:
        return ""

# ── Cognee index ──────────────────────────────────────────────────────────────

def _load_chunks(path: str) -> list[str]:
    if path.endswith(".pdf"):
        reader = PdfReader(path)
        chunks = []
        for page in reader.pages:
            text = page.extract_text() or ""
            for para in text.split("\n\n"):
                para = para.strip()
                if len(para) > 80:
                    chunks.append(para)
        return chunks
    text = open(path).read()
    return [p.strip() for p in text.split("\n\n") if p.strip()]

@st.cache_resource
def build_index(_tmp_path: str, file_key: str):
    async def _build():
        await cognee.forget(everything=True)
        for chunk in _load_chunks(_tmp_path):
            await cognee.remember(chunk)

    asyncio.run(_build())

# ── RAG pipeline ──────────────────────────────────────────────────────────────

def retrieve(query: str) -> list[str]:
    results = asyncio.run(cognee.recall(query))
    texts = []
    for r in results[:3]:
        text = r.text if hasattr(r, "text") else str(r)
        texts.append(text)
    return texts

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

def rag(query: str) -> str:
    chunks = retrieve(query)
    prompt = augment(query, chunks)
    return generate(prompt)

# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.title("RAG Chatbot (Cognee)")

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

with st.spinner("Building knowledge graph..."):
    build_index(st.session_state["tmp_path"], file_key)

st.caption(f"Knowledge base: `{uploaded.name}`")

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
            answer = rag(query)
        st.write(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
