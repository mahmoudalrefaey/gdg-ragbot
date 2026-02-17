"""
app.py — Streamlit RAG chat. Run from rag_bot: streamlit run app.py
Requires: pip install -r requirements.txt, ollama pull llama3
"""

import time
import json
import urllib.request
import urllib.error
import streamlit as st
import rag

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 RAG Chatbot")
st.caption("Ask about documents in data/. Uses Chroma + Ollama.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Chunks to retrieve", 1, 10, 4)
    show_debug = st.toggle("Show retrieved context", False)
    st.divider()
    if st.button("🔄 Rebuild Index"):
        with st.spinner("Rebuilding ..."):
            coll = rag.build_index()
            st.session_state["collection"] = coll
            st.success("Index rebuilt.")


@st.cache_resource(show_spinner="Loading Chroma ...")
def init_collection():
    return rag.get_collection()

if "collection" not in st.session_state:
    st.session_state["collection"] = init_collection()


def ask_ollama(prompt):
    """Call Ollama with prompt. Returns (answer, time_ms)."""
    t0 = time.perf_counter()
    body = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 512},
    }).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode())
    ms = (time.perf_counter() - t0) * 1000
    return out["response"].strip(), ms


if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask something ..."):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking ..."):
            try:
                results, ret_ms = rag.retrieve(
                    question, st.session_state["collection"], top_k=top_k
                )
                prompt = rag.build_prompt(question, results)
                answer, gen_ms = ask_ollama(prompt)
                st.markdown(answer)

                if show_debug:
                    with st.expander("📋 Retrieved chunks", expanded=False):
                        st.caption(f"Retrieval: {ret_ms:.0f} ms  |  Generation: {gen_ms:.0f} ms")
                        for i, c in enumerate(results, 1):
                            st.markdown(f"**{i}** — {c['source']} #{c['chunk_id']} (dist: {c['score']:.4f})")
                            st.code(c["text"][:400], language=None)

                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except urllib.error.URLError as e:
                st.error("Cannot reach Ollama. Run: ollama pull llama3")
            except Exception as e:
                st.error(str(e))
