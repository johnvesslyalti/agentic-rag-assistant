# Agentic RAG Assistant 🤖

Conversational AI agent that decides when to retrieve, search, or answer directly — built on LangGraph, FAISS, and OpenAI.

## What it does

Unlike a standard RAG pipeline that always retrieves, this agent **reasons** about each query and picks the right tool:

- `retrieve_essays` — searches Paul Graham's essays via FAISS vector search
- `web_search` — fetches live info via Tavily when essays aren't enough
- `answer_directly` — responds from LLM knowledge when no retrieval is needed

It also maintains **conversation memory** across turns and streams responses via a FastAPI backend.

## Architecture

```
User query
    │
    ▼
LangGraph ReAct Agent
    ├── retrieve_essays  (FAISS + OpenAI embeddings)
    ├── web_search       (Tavily)
    └── answer_directly  (GPT-4o)
    │
    ▼
FastAPI /chat endpoint  (streaming SSE)
    │
    ▼
Streamlit Chat UI
```

## Stack

| Layer | Tool |
|---|---|
| Agent framework | LangGraph |
| Retrieval | FAISS + LangChain |
| LLM | OpenAI GPT-4o |
| Web search | Tavily |
| API | FastAPI + streaming |
| UI | Streamlit |
| Deployment | Railway + Vercel |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your keys to `.env`:
```
OPENAI_API_KEY=your_openai_key
TAVILY_API_KEY=your_tavily_key
```

## Build the FAISS index

Run once to download Paul Graham essays and create the vector index:

```bash
python scripts/ingest.py
```

This fetches 14 essays from paulgraham.com, chunks them, embeds via OpenAI, and saves the index to `retriever/faiss_index/`.

Verify retrieval works:
```bash
python retriever/retriever.py
```

## Run

**API only:**
```bash
uvicorn api.main:app --reload
```

**Full stack with Docker Compose:**
```bash
docker compose up --build
# API → http://localhost:8000
# UI  → http://localhost:8501
```

**UI only (pointing at a running API):**
```bash
streamlit run ui/app.py
```

## Test

```bash
pytest
```

Tests run without API keys — all LLM calls are mocked.

## Tool isolation test

```bash
python tools/test_tools.py
```

API-dependent tools skip gracefully when keys are absent.

## Built by

[@zavxai](https://x.com/zavxai) — extends [PG RAG Engine](https://github.com/johnvesslyalti/pg-rag-engine) with agentic reasoning.
