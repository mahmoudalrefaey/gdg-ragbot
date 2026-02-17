"""
rag.py — Simple RAG using Chroma (a real vector database).

Functions:
  load_documents()  — read .txt/.md from data/
  chunk_text()      — split into overlapping pieces
  build_index()     — embed chunks, store in Chroma
  get_collection()  — get Chroma collection (builds if missing)
  retrieve()        — search Chroma for top-k chunks
  build_prompt()    — context + question for the LLM
"""

import os
import time
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Simple paths: run from rag_bot folder, so "data" and "chroma_db" work
DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

log.info("Loading embedding model ...")
model = SentenceTransformer(EMBED_MODEL)
log.info("Model loaded.")


def load_documents():
    """Read every .txt and .md file in data/. Returns list of {source, text}."""
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Folder not found: {DATA_DIR}")

    docs = []
    for name in sorted(os.listdir(DATA_DIR)):
        if name.endswith(".txt") or name.endswith(".md"):
            path = os.path.join(DATA_DIR, name)
            with open(path, encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                docs.append({"source": name, "text": text})
                log.info("  Loaded %s", name)

    if not docs:
        raise ValueError(f"No .txt/.md files in {DATA_DIR}")
    return docs


def chunk_text(text, source):
    """Split text into overlapping chunks. Returns list of {text, source, chunk_id}."""
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        piece = text[start : start + CHUNK_SIZE].strip()
        if piece:
            chunks.append({"text": piece, "source": source, "chunk_id": idx})
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def get_client():
    """Chroma client that persists to chroma_db/."""
    return chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(anonymized_telemetry=False))


def build_index():
    """Load docs, chunk, embed, and store in Chroma. Returns the collection."""
    log.info("Building index ...")

    docs = load_documents()
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc["text"], doc["source"]))
    log.info("Chunks: %d", len(chunks))

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    client = get_client()
    try:
        client.delete_collection("rag")
    except Exception:
        pass
    coll = client.create_collection("rag", metadata={"description": "RAG chunks"})

    ids = [f"{c['source']}_{c['chunk_id']}" for c in chunks]
    metadatas = [{"source": c["source"], "chunk_id": c["chunk_id"]} for c in chunks]
    coll.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    log.info("Chroma index saved to %s", CHROMA_DIR)
    return coll


def get_collection():
    """Get Chroma collection. Build index if it doesn't exist."""
    client = get_client()
    try:
        return client.get_collection("rag")
    except Exception:
        return build_index()


def retrieve(query, collection, top_k=4):
    """
    Embed query, search Chroma, return list of {text, source, chunk_id, score}.
    score = distance (lower is more similar). Returns (results, time_ms).
    """
    t0 = time.perf_counter()
    q_embedding = model.encode([query]).tolist()
    out = collection.query(query_embeddings=q_embedding, n_results=top_k, include=["documents", "metadatas", "distances"])

    results = []
    if out["documents"] and out["documents"][0]:
        for doc, meta, dist in zip(out["documents"][0], out["metadatas"][0], out["distances"][0]):
            results.append({
                "text": doc,
                "source": meta["source"],
                "chunk_id": meta["chunk_id"],
                "score": round(float(dist), 4),
            })
    ms = (time.perf_counter() - t0) * 1000
    log.info("Retrieved %d in %.0f ms", len(results), ms)
    return results, ms


def build_prompt(question, chunks):
    """Build prompt: system + context + question. Sources in context."""
    if not chunks:
        return f"Question: {question}\n\nAnswer (say you have no context):"

    context = "\n\n".join(
        f"[Source: {c['source']} | Chunk #{c['chunk_id']}]\n{c['text']}" for c in chunks
    )
    sources = ", ".join(sorted(set(c["source"] for c in chunks)))
    return (
        "Answer only from the context below. If not in context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (end with Sources: {sources}):"
    )
