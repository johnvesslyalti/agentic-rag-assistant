# Agentic RAG Assistant 🤖

[![CI](https://github.com/johnvesslyalti/agentic-rag-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/johnvesslyalti/agentic-rag-assistant/actions/workflows/ci.yml)

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
| API deployment | Railway |
| UI deployment | Streamlit Community Cloud |

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

## Deploy

### API → Railway

1. Push this repo to GitHub.
2. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
3. Select this repository. Railway auto-detects `railway.toml` and builds from `Dockerfile`.
4. Add environment variables in the Railway dashboard:
   - `OPENAI_API_KEY`
   - `TAVILY_API_KEY`
5. Upload your FAISS index as a Railway volume mounted at `/app/retriever/faiss_index`, **or** run `scripts/ingest.py` as a one-off job inside the container before starting the API.
6. Railway exposes the service on a public URL — note it for the UI step.

### UI → Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app** → select this repository → set **Main file path** to `ui/app.py`.
3. Under **Advanced settings → Secrets**, add:
   ```toml
   OPENAI_API_KEY = "your_openai_key"
   TAVILY_API_KEY = "your_tavily_key"
   API_URL = "https://your-railway-url.railway.app"
   ```
4. Deploy — Streamlit reads `API_URL` from environment at startup (see sidebar input in `ui/app.py`).

### Full stack locally (Docker Compose)

```bash
cp .env.example .env   # fill in your keys
docker compose up --build
# API → http://localhost:8000
# UI  → http://localhost:8501
```

---

## Built by

[@zavxai](https://x.com/zavxai) — extends [PG RAG Engine](https://github.com/johnvesslyalti/pg-rag-engine) with agentic reasoning.
