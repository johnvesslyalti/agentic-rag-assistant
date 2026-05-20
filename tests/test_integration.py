"""
Integration tests — FastAPI runs for real, but without OpenAI/Tavily keys.
All tools return graceful error strings; no mocks involved.

Run: pytest tests/test_integration.py -v
"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def real_client():
    """Client backed by the real app and real (uncached) agent."""
    import agent.agent as mod
    mod._agent = None
    from api.main import app
    with TestClient(app) as c:
        yield c
    mod._agent = None


def test_real_health(real_client):
    resp = real_client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["retriever"] in ("ready", "index missing — run scripts/ingest.py")


def test_real_root(real_client):
    resp = real_client.get("/")
    assert resp.status_code == 200
    assert resp.json()["version"] == "1.0.0"


def test_real_chat_validation_empty_query(real_client):
    resp = real_client.post("/chat", json={"query": "", "session_id": "int-1"})
    assert resp.status_code == 422


def test_real_chat_validation_too_long(real_client):
    resp = real_client.post("/chat", json={"query": "q" * 2001, "session_id": "int-2"})
    assert resp.status_code == 422


@pytest.mark.skipif(
    not __import__("os").getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live agent test",
)
def test_real_stream_endpoint_reachable(real_client):
    """Stream endpoint returns SSE when API key is available."""
    resp = real_client.post(
        "/chat/stream",
        json={"query": "Hello", "session_id": "int-stream"},
    )
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    assert "data:" in resp.text
