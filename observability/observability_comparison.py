#!/usr/bin/env python3
"""
Side-by-side comparison of Logfire vs Arize observability,
both using FAISS RAG as the pipeline.

Run: streamlit run rag/observability_comparison.py
"""

import os
import tempfile
import textwrap
from dotenv import load_dotenv

load_dotenv()

import logfire
from pypdf import PdfReader
from openai import OpenAI
import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from arize.otel import register as _arize_register, Transport

CHAT_MODEL   = "grok-3-mini"
PROFILE_FILE = "suraj_profile_summary.txt"
ARIZE_URL    = "https://app.arize.com"

_grok = OpenAI(api_key=os.getenv("XAI_API_KEY", ""), base_url="https://api.x.ai/v1")

# OpenInference attribute names Arize uses to populate Input/Output columns
_OI_SPAN_KIND = "openinference.span.kind"
_OI_INPUT     = "input.value"
_OI_OUTPUT    = "output.value"

# ── Observability setup — cached so Streamlit re-runs don't re-configure ──────

@st.cache_resource
def _setup_logfire():
    logfire.configure()

@st.cache_resource
def _arize_tracer():
    provider = _arize_register(
        space_id=os.getenv("ARIZE_SPACE_ID", ""),
        api_key=os.getenv("ARIZE_API_KEY", ""),
        project_name="faiss-rag-observability",
        transport=Transport.GRPC,
        batch=False,
        set_global_tracer_provider=False,
        verbose=False,
    )
    return provider.get_tracer("faiss-rag")

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

def _call_grok(prompt: str) -> str:
    profile = load_profile(PROFILE_FILE)
    messages = []
    if profile:
        messages.append({"role": "system", "content": f"User profile:\n{profile}"})
    messages.append({"role": "user", "content": prompt})
    return _grok.chat.completions.create(model=CHAT_MODEL, messages=messages).choices[0].message.content

# ── FAISS index (shared between both columns) ─────────────────────────────────

class VectorStore:
    def build(self, docs: list[dict], model: SentenceTransformer) -> None:
        self._docs = docs
        self._model = model
        vecs = self._embed([d["text"] for d in docs])
        self._index = faiss.IndexFlatIP(vecs.shape[1])
        self._index.add(vecs)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        q = self._embed([query])
        _, indices = self._index.search(q, top_k)
        return [self._docs[i]["text"] for i in indices[0] if i != -1]

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(vecs)
        return vecs

@st.cache_resource
def build_faiss_index(_tmp_path: str, file_key: str) -> VectorStore:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = load_docs(_tmp_path)
    store = VectorStore()
    store.build(docs, model)
    return store

# ── Logfire-instrumented RAG ──────────────────────────────────────────────────

def logfire_rag(query: str, store: VectorStore) -> str:
    with logfire.span("faiss.rag", query=query) as root:
        with logfire.span("faiss.retrieve"):
            chunks = store.search(query)
        logfire.info("faiss.retrieved", chunk_count=len(chunks))
        with logfire.span("llm.generate", model=CHAT_MODEL, provider="xai"):
            answer = _call_grok(augment(query, chunks))
        root.set_attribute("answer", answer)
    return answer

# ── Arize-instrumented RAG ────────────────────────────────────────────────────

def arize_rag(query: str, store: VectorStore, tracer) -> str:
    with tracer.start_as_current_span("faiss.rag") as span:
        span.set_attribute(_OI_SPAN_KIND, "CHAIN")
        span.set_attribute(_OI_INPUT, query)

        with tracer.start_as_current_span("faiss.retrieve") as ret_span:
            ret_span.set_attribute(_OI_SPAN_KIND, "RETRIEVER")
            ret_span.set_attribute(_OI_INPUT, query)
            chunks = store.search(query)
            for i, chunk in enumerate(chunks):
                ret_span.set_attribute(f"retrieval.documents.{i}.document.content", chunk)

        with tracer.start_as_current_span("llm.generate") as gen_span:
            gen_span.set_attribute(_OI_SPAN_KIND, "LLM")
            gen_span.set_attribute("llm.model_name", CHAT_MODEL)
            gen_span.set_attribute(_OI_INPUT, augment(query, chunks))
            answer = _call_grok(augment(query, chunks))
            gen_span.set_attribute(_OI_OUTPUT, answer)

        span.set_attribute(_OI_OUTPUT, answer)
    return answer

# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(layout="wide")
st.title("Observability Comparison: Logfire vs Arize")
st.caption("Same FAISS RAG pipeline — different observability backends.")

_setup_logfire()
ph_tracer = _arize_tracer()

col_lf, col_ph = st.columns(2)
col_lf.subheader("Logfire")
col_lf.markdown("[Open Logfire dashboard →](https://logfire.pydantic.dev/)")
col_ph.subheader("Arize")
col_ph.markdown(f"[Open Arize dashboard →]({ARIZE_URL})")

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

with st.spinner("Building FAISS index..."):
    store = build_faiss_index(st.session_state["tmp_path"], file_key)

st.caption(f"Knowledge base: `{uploaded.name}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    col = col_lf if msg["col"] == "logfire" else col_ph
    with col.chat_message(msg["role"]):
        col.write(msg["content"])

if query := st.chat_input("Ask a question..."):
    logfire.info("user.query", query=query)

    col_lf.chat_message("user").write(query)
    col_ph.chat_message("user").write(query)
    st.session_state.messages.append({"role": "user", "content": query, "col": "logfire"})
    st.session_state.messages.append({"role": "user", "content": query, "col": "phoenix"})

    with col_lf.chat_message("assistant"):
        with st.spinner("Thinking..."):
            lf_answer = logfire_rag(query, store)
        col_lf.write(lf_answer)

    logfire.force_flush()

    with col_ph.chat_message("assistant"):
        with st.spinner("Thinking..."):
            ph_answer = arize_rag(query, store, ph_tracer)
        col_ph.write(ph_answer)

    st.session_state.messages.append({"role": "assistant", "content": lf_answer,  "col": "logfire"})
    st.session_state.messages.append({"role": "assistant", "content": ph_answer, "col": "phoenix"})
