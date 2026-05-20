"""
FastAPI endpoint tests — no API keys required, all LLM calls are mocked.
Run: pytest tests/test_api.py -v
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


def _make_mock_agent():
    agent = MagicMock()
    agent.ainvoke = AsyncMock(
        return_value={"messages": [AIMessage(content="Mock LLM response")]}
    )

    async def _astream_events(state, config, version):
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="Mock ")}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": MagicMock(content="response")}}

    agent.astream_events = _astream_events
    return agent


@pytest.fixture()
def client():
    with patch("api.main.get_agent", return_value=_make_mock_agent()):
        from api.main import app
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Root and health
# ---------------------------------------------------------------------------

def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Agentic RAG Assistant"
    assert "endpoints" in data


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "retriever" in data


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

def test_chat_returns_response(client):
    resp = client.post("/chat", json={"query": "What is 2 + 2?", "session_id": "t1"})
    assert resp.status_code == 200
    data = resp.json()
    assert "response" in data
    assert data["session_id"] == "t1"


def test_chat_default_session_id(client):
    resp = client.post("/chat", json={"query": "Hello"})
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "default"


def test_chat_missing_query_returns_422(client):
    resp = client.post("/chat", json={"session_id": "t2"})
    assert resp.status_code == 422


def test_chat_empty_body_returns_422(client):
    resp = client.post("/chat", json={})
    assert resp.status_code == 422


def test_chat_query_too_long_returns_422(client):
    resp = client.post("/chat", json={"query": "x" * 2001, "session_id": "t-long"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /chat/stream
# ---------------------------------------------------------------------------

def test_chat_stream_returns_sse(client):
    resp = client.post("/chat/stream", json={"query": "Hello", "session_id": "t3"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]


def test_chat_stream_contains_data_lines(client):
    resp = client.post("/chat/stream", json={"query": "Hello", "session_id": "t4"})
    body = resp.text
    assert "data:" in body
