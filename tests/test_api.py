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
    assert "history" in data["endpoints"]


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


# ---------------------------------------------------------------------------
# Session history
# ---------------------------------------------------------------------------

def test_session_history_empty_for_new_session(client):
    resp = client.get("/sessions/never-used-session/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "never-used-session"
    assert data["messages"] == []
    assert data["count"] == 0


def test_session_history_records_chat_turns(client):
    sid = "history-test-session"
    client.post("/chat", json={"query": "What is 2+2?", "session_id": sid})
    resp = client.get(f"/sessions/{sid}/history")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 2
    roles = [m["role"] for m in data["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_session_history_contains_query_text(client):
    sid = "history-content-test"
    query = "Tell me about Paul Graham essays"
    client.post("/chat", json={"query": query, "session_id": sid})
    resp = client.get(f"/sessions/{sid}/history")
    messages = resp.json()["messages"]
    user_msgs = [m for m in messages if m["role"] == "user"]
    assert any(query in m["content"] for m in user_msgs)


# ---------------------------------------------------------------------------
# GET /sessions
# ---------------------------------------------------------------------------

def test_list_sessions_returns_list(client):
    resp = client.get("/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert "sessions" in data
    assert "count" in data
    assert isinstance(data["sessions"], list)


def test_list_sessions_includes_used_session(client):
    sid = "list-sessions-test"
    client.post("/chat", json={"query": "Hello", "session_id": sid})
    resp = client.get("/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert sid in data["sessions"]
    assert data["count"] == len(data["sessions"])


# ---------------------------------------------------------------------------
# DELETE /sessions/{session_id}/history
# ---------------------------------------------------------------------------

def test_delete_session_history_returns_204(client):
    sid = "delete-test-session"
    client.post("/chat", json={"query": "Hello", "session_id": sid})
    resp = client.delete(f"/sessions/{sid}/history")
    assert resp.status_code == 204


def test_delete_session_history_clears_messages(client):
    sid = "delete-clear-test"
    client.post("/chat", json={"query": "Remember this", "session_id": sid})
    history_before = client.get(f"/sessions/{sid}/history").json()
    assert history_before["count"] > 0

    client.delete(f"/sessions/{sid}/history")

    history_after = client.get(f"/sessions/{sid}/history").json()
    assert history_after["count"] == 0
    assert history_after["messages"] == []


def test_delete_nonexistent_session_returns_204(client):
    """Deleting a session that never existed is idempotent."""
    resp = client.delete("/sessions/never-existed-xyz/history")
    assert resp.status_code == 204


def test_list_sessions_excludes_deleted_session(client):
    sid = "delete-exclude-test"
    client.post("/chat", json={"query": "Hello", "session_id": sid})
    client.delete(f"/sessions/{sid}/history")
    resp = client.get("/sessions")
    assert sid not in resp.json()["sessions"]


# ---------------------------------------------------------------------------
# Root endpoint advertises new endpoints
# ---------------------------------------------------------------------------

def test_root_advertises_sessions_endpoint(client):
    resp = client.get("/")
    data = resp.json()
    assert "sessions" in data["endpoints"]
    assert "delete_history" in data["endpoints"]


# ---------------------------------------------------------------------------
# Source citation format
# ---------------------------------------------------------------------------

def test_retrieve_essays_tool_includes_source_label():
    """retrieve_essays should prefix each chunk with [Source: <title>] when index is missing it returns an error string, not an exception."""
    from tools.tools import retrieve_essays
    result = retrieve_essays.invoke("What does Paul Graham say about startups?")
    assert isinstance(result, str)
    assert len(result) > 0
    # Either a graceful error OR a citation-formatted response
    has_citation = "[Source:" in result
    has_error = "unavailable" in result or "error" in result.lower()
    assert has_citation or has_error, f"Unexpected result: {result[:120]}"
