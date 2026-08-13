---
title: Qwen RAG Chatbot
emoji: 🧠
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Qwen RAG Chatbot

A Docker-ready Retrieval-Augmented Generation chatbot using FastAPI, Hugging Face Inference API, Qwen/Qwen2.5-3B-Instruct, Sentence Transformers, FAISS, and a Monaco-based HackerRank-inspired frontend.

## Run locally

```bash
python -m venv .venv
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860.

## Docker

```bash
docker build -t qwen-rag-chatbot .
docker run --rm -p 7860:7860 -e HF_TOKEN="your_token" qwen-rag-chatbot
```

## Hugging Face Spaces

Create a Docker Space and add `HF_TOKEN` as a Space Secret. The README metadata configures Docker and port 7860.

The frontend also allows an optional session HF token. Token precedence is:

1. Session token supplied by the user
2. `HF_TOKEN` environment variable / Space Secret
3. Local Transformers fallback

Supported files: PDF, TXT, MD.
