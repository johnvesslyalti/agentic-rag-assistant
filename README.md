# Agentic RAG Assistant 🤖

Conversational AI agent that decides when to retrieve, search, or answer directly — built on LangGraph, FAISS, and OpenAI.

## What it does

Unlike a standard RAG pipeline that always retrieves, this agent **reasons** about each query and picks the right tool:

- `retrieve_essays` — searches Paul Graham's essays via FAISS
- `web_search` — fetches live info via Tavily when essays aren't enough
- `answer_directly` — responds from LLM knowledge when no retrieval is needed

It also maintains **conversation memory** across turns and streams responses via a FastAPI backend.

## Architecture

```
User query
    │
    ▼
LangGraph ReAct Agent
    ├── retrieve_essays (FAISS)
    ├── web_search (Tavily)
    └── answer_directly (LLM)
    │
    ▼
FastAPI /chat endpoint (streaming)
    │
    ▼
Chat UI (Streamlit / Next.js)
```

## Stack

| Layer | Tool |
|---|---|
| Agent framework | LangGraph |
| Retrieval | FAISS + LlamaIndex |
| LLM | OpenAI GPT-4o |
| Web search | Tavily |
| API | FastAPI + streaming |
| Deployment | Railway + Vercel |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your keys to `.env`:
```
OPENAI_API_KEY=your_key
TAVILY_API_KEY=your_key
```

Place your FAISS index inside `retriever/faiss_index/`.

## Run

```bash
uvicorn api.main:app --reload
```

## Built by

[@zavxai](https://x.com/zavxai) — extends [PG RAG Engine](https://github.com/johnvesslyalti/pg-rag-engine) with agentic reasoning.
