# RAG Chatbot

A minimal RAG (Retrieval-Augmented Generation) chatbot: ask questions about your documents. Uses **Chroma** as the vector database and **Ollama** for the LLM. No API keys, no paid services.

---

## Project structure

```
rag_bot/
├── app.py           # Streamlit chat UI
├── rag.py           # RAG logic: load, chunk, embed, Chroma, retrieve
├── data/            # Put your .txt and .md files here
│   ├── sample.txt
│   └── sample2.md
├── chroma_db/       # Created automatically (Chroma store)
├── requirements.txt
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **Ollama** — [Download](https://ollama.com), then run: `ollama pull llama3`

---

## Setup

1. **Install dependencies** (use the same Python you will use to run the app):

   ```bash
   cd rag_bot
   pip install -r requirements.txt
   ```

   On Windows, if you use a specific Python (e.g. 3.13):

   ```bash
   py -3.13 -m pip install -r requirements.txt
   ```

2. **Run the app**:

   ```bash
   streamlit run app.py
   ```

3. Open **http://localhost:8501** in your browser.

---

## Usage

- Type a question in the chat. The app retrieves relevant chunks from your documents and asks Ollama to answer using that context.
- **Sidebar:** Adjust “Chunks to retrieve” (top_k). Turn on “Show retrieved context” to see which chunks were used and their similarity scores.
- **Rebuild Index:** Click when you add or change files in `data/`. This rebuilds the Chroma index from all .txt and .md files in `data/`.

---

## How it works

1. **Documents** in `data/` are split into overlapping chunks (~600 chars, 100 char overlap).
2. **Chroma** stores embeddings of those chunks (from SentenceTransformers `all-MiniLM-L6-v2`).
3. When you **ask a question**, it is embedded and Chroma returns the most similar chunks.
4. Those chunks plus your question are sent to **Ollama** (llama3), which answers using only that context.

So the answer is grounded in your documents instead of the model’s training data (fewer hallucinations).

---

## Adding your own documents

1. Add `.txt` or `.md` files to the `data/` folder.
2. In the app sidebar, click **🔄 Rebuild Index**.
3. Ask questions as usual.

---

## Troubleshooting

| Issue | What to do |
|--------|------------|
| `ModuleNotFoundError: No module named 'chromadb'` (or similar) | You have more than one Python. Install with the one that runs Streamlit, e.g. `py -3.13 -m pip install -r requirements.txt`, then run the app with that same Python. |
| Cannot reach Ollama | Install Ollama from https://ollama.com, run `ollama pull llama3`, and ensure Ollama is running (it often starts with the app). |
| No .txt or .md files found | Add at least one `.txt` or `.md` file with content inside `data/`. |
| Slow first run | The first time you run the app, the embedding model is downloaded (~80 MB). Later runs use the cached model and Chroma index. |

---

## Tech stack

- **Streamlit** — chat UI
- **Chroma** — vector database (persists in `chroma_db/`)
- **SentenceTransformers** — embeddings (`all-MiniLM-L6-v2`)
- **Ollama** — local LLM (llama3); called via `urllib`
