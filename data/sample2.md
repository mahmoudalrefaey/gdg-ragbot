# Frequently Asked Questions about AI Chatbots

## What is an LLM?
A Large Language Model (LLM) is a deep learning model trained on massive amounts of text data. Examples include GPT-4, LLaMA 3, Mistral, and Gemma. LLMs predict the next token (word/subword) in a sequence, which allows them to generate fluent, human-like text. However, they can only "know" what was in their training data, which is why RAG is so valuable — it gives them access to fresh, domain-specific information at query time.

## What is FAISS?
FAISS stands for Facebook AI Similarity Search. It is an open-source library developed by Meta AI for efficient similarity search and clustering of dense vectors. FAISS is used in production systems that need to search through billions of vectors quickly. In our project, we use FAISS as a local vector store — no cloud service or API key needed.

## What is Ollama?
Ollama is a tool that lets you run open-source LLMs locally on your own machine. It supports models like LLaMA 3, Mistral, Phi-3, and Gemma. Running models locally means you don't need API keys or internet access, and your data stays private. To install Ollama, visit https://ollama.com and follow the instructions for your operating system.

## How to install Ollama on Windows?
1. Go to https://ollama.com/download
2. Download the Windows installer
3. Run the installer and follow the prompts
4. Open a terminal and run: ollama pull llama3
5. Wait for the model to download (about 4.7 GB)
6. Test it: ollama run llama3 "Hello, how are you?"

## What is SentenceTransformers?
SentenceTransformers is a Python library that provides easy-to-use pre-trained models for generating sentence and text embeddings. The model "all-MiniLM-L6-v2" is a popular choice because it is small (80 MB), fast, and produces 384-dimensional embeddings that work well for semantic search tasks.

## Why not use LangChain?
LangChain is a great framework for building LLM applications, but it abstracts away many details. For learning purposes, building RAG from scratch helps you understand what happens at each step. Once you understand the fundamentals, you can use LangChain or LlamaIndex to move faster in production.
