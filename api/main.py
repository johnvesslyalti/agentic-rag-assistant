import asyncio
import json
import logging
import os
import time
from collections import defaultdict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("rag_api")

from agent.agent import get_agent

app = FastAPI(
    title="Agentic RAG Assistant",
    version="1.0.0",
    description="ReAct agent with FAISS retrieval, Tavily web search, and streaming.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Rate limiting — configurable via RATE_LIMIT_PER_MIN env var (0 = disabled)
# ---------------------------------------------------------------------------
_RATE_LIMIT = int(os.getenv("RATE_LIMIT_PER_MIN", "60"))
_ip_request_times: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if _RATE_LIMIT <= 0:
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    window_start = now - 60.0

    bucket = _ip_request_times[client_ip]
    # Evict timestamps outside the 1-minute window
    _ip_request_times[client_ip] = [t for t in bucket if t > window_start]

    if len(_ip_request_times[client_ip]) >= _RATE_LIMIT:
        logger.warning("rate_limit ip=%s requests=%d", client_ip, len(_ip_request_times[client_ip]))
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Try again in a minute."},
        )

    _ip_request_times[client_ip].append(now)
    return await call_next(request)


# In-memory session history store: {session_id: [{"role": ..., "content": ...}]}
_session_histories: dict[str, list[dict]] = {}


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]
    count: int


class SessionsListResponse(BaseModel):
    sessions: list[str]
    count: int


@app.get("/sessions", response_model=SessionsListResponse)
async def list_sessions():
    """Return all session IDs that have stored history."""
    sessions = list(_session_histories.keys())
    return SessionsListResponse(sessions=sessions, count=len(sessions))


@app.delete("/sessions/{session_id}/history", status_code=204)
async def delete_session_history(session_id: str):
    """Clear the stored conversation history for a session."""
    _session_histories.pop(session_id, None)


@app.get("/")
async def root():
    return {
        "name": "Agentic RAG Assistant",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "chat": "POST /chat",
            "stream": "POST /chat/stream",
            "sessions": "GET /sessions",
            "history": "GET /sessions/{session_id}/history",
            "delete_history": "DELETE /sessions/{session_id}/history",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
async def health():
    index_path = os.path.join(
        os.path.dirname(__file__), "..", "retriever", "faiss_index"
    )
    retriever_ready = os.path.exists(os.path.abspath(index_path))
    return {
        "status": "ok",
        "retriever": "ready" if retriever_ready else "index missing — run scripts/ingest.py",
    }


@app.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """Return the stored conversation history for a session."""
    messages = _session_histories.get(session_id, [])
    return SessionHistoryResponse(
        session_id=session_id,
        messages=messages,
        count=len(messages),
    )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream tokens from the agent as server-sent events."""
    agent = get_agent()
    config = {"configurable": {"thread_id": request.session_id}}
    input_state = {"messages": [HumanMessage(content=request.query)]}

    _session_histories.setdefault(request.session_id, []).append(
        {"role": "user", "content": request.query}
    )
    logger.info("stream_start session=%s query_len=%d", request.session_id, len(request.query))
    t0 = time.monotonic()

    async def generate():
        full_response = ""
        try:
            async for event in agent.astream_events(input_state, config=config, version="v2"):
                kind = event["event"]
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        full_response += chunk.content
                        yield f"data: {json.dumps({'token': chunk.content})}\n\n"
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "")
                    if tool_name:
                        logger.info(
                            "tool_start session=%s tool=%s", request.session_id, tool_name
                        )
                        yield f"data: {json.dumps({'tool_start': tool_name})}\n\n"
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "")
                    if tool_name:
                        yield f"data: {json.dumps({'tool_end': tool_name})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error("stream_error session=%s error=%s", request.session_id, str(e))
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            elapsed = time.monotonic() - t0
            if full_response:
                _session_histories.setdefault(request.session_id, []).append(
                    {"role": "assistant", "content": full_response}
                )
            logger.info(
                "stream_done session=%s elapsed=%.2fs response_len=%d",
                request.session_id, elapsed, len(full_response),
            )

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint; returns the full response once complete."""
    agent = get_agent()
    config = {"configurable": {"thread_id": request.session_id}}
    input_state = {"messages": [HumanMessage(content=request.query)]}

    _session_histories.setdefault(request.session_id, []).append(
        {"role": "user", "content": request.query}
    )
    logger.info("chat_start session=%s query_len=%d", request.session_id, len(request.query))
    t0 = time.monotonic()

    _timeout = float(os.getenv("AGENT_TIMEOUT", "120"))
    try:
        result = await asyncio.wait_for(
            agent.ainvoke(input_state, config=config),
            timeout=_timeout,
        )
        last_message = result["messages"][-1]
        response_text = last_message.content
        _session_histories.setdefault(request.session_id, []).append(
            {"role": "assistant", "content": response_text}
        )
        elapsed = time.monotonic() - t0
        logger.info(
            "chat_done session=%s elapsed=%.2fs response_len=%d",
            request.session_id, elapsed, len(response_text),
        )
        return ChatResponse(response=response_text, session_id=request.session_id)
    except asyncio.TimeoutError:
        logger.error("chat_timeout session=%s timeout=%.0fs", request.session_id, _timeout)
        raise HTTPException(status_code=504, detail=f"Agent did not respond within {_timeout:.0f}s")
    except Exception as e:
        logger.error("chat_error session=%s error=%s", request.session_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))
