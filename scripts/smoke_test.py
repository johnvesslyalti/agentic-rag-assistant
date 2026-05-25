#!/usr/bin/env python3
"""
Smoke test — verify a running API instance responds correctly on all key endpoints.

Usage:
    python scripts/smoke_test.py                          # defaults to http://localhost:8000
    python scripts/smoke_test.py http://your-railway-url.railway.app

Exit code 0 = all checks passed, 1 = one or more failures.
The /chat/stream check requires OPENAI_API_KEY and is skipped otherwise.
"""
import json
import os
import sys
import uuid

import requests

BASE_URL = (sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000")
HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
SESSION_ID = f"smoke-{uuid.uuid4().hex[:8]}"
TIMEOUT = 15
_failures: list[str] = []


def _check(label: str, resp: requests.Response, expected: int = 200) -> bool:
    ok = resp.status_code == expected
    status = "OK  " if ok else "FAIL"
    print(f"  {status}  {label}  (HTTP {resp.status_code})")
    if not ok:
        print(f"        body: {resp.text[:200]}")
        _failures.append(label)
    return ok


def run() -> bool:
    s = requests.Session()

    # GET /health
    r = s.get(f"{BASE_URL}/health", timeout=TIMEOUT)
    if _check("GET /health", r):
        data = r.json()
        assert data["status"] == "ok", f"Expected status=ok, got {data}"
        print(f"        retriever: {data.get('retriever')}")

    # GET /
    r = s.get(f"{BASE_URL}/", timeout=TIMEOUT)
    if _check("GET /", r):
        assert "endpoints" in r.json()

    # GET /docs  (Swagger UI)
    _check("GET /docs", s.get(f"{BASE_URL}/docs", timeout=TIMEOUT))

    # POST /chat — validation: empty query must be rejected
    r = s.post(f"{BASE_URL}/chat", json={"query": "", "session_id": SESSION_ID}, timeout=TIMEOUT)
    _check("POST /chat (empty query → 422)", r, expected=422)

    # POST /chat — validation: query too long must be rejected
    r = s.post(f"{BASE_URL}/chat", json={"query": "x" * 2001, "session_id": SESSION_ID}, timeout=TIMEOUT)
    _check("POST /chat (too long → 422)", r, expected=422)

    # GET /sessions/{id}/history — new session has empty history
    r = s.get(f"{BASE_URL}/sessions/{SESSION_ID}/history", timeout=TIMEOUT)
    if _check("GET /sessions/{id}/history (empty)", r):
        data = r.json()
        assert data["count"] == 0, f"Expected count=0 for new session, got {data['count']}"

    # GET /sessions — list endpoint exists
    r = s.get(f"{BASE_URL}/sessions", timeout=TIMEOUT)
    if _check("GET /sessions", r):
        data = r.json()
        assert isinstance(data["sessions"], list)

    # DELETE /sessions/{id}/history — idempotent even for unknown session
    r = s.delete(f"{BASE_URL}/sessions/{SESSION_ID}/history", timeout=TIMEOUT)
    _check("DELETE /sessions/{id}/history (idempotent)", r, expected=204)

    # POST /chat/stream — requires OPENAI_API_KEY; skip gracefully if absent
    if not HAS_OPENAI:
        print("  SKIP  POST /chat/stream (OPENAI_API_KEY not set)")
    else:
        r = s.post(
            f"{BASE_URL}/chat/stream",
            json={"query": "Hello", "session_id": SESSION_ID},
            timeout=TIMEOUT,
            stream=True,
        )
        if _check("POST /chat/stream (SSE headers)", r):
            assert "text/event-stream" in r.headers.get("content-type", ""), (
                f"Expected text/event-stream, got {r.headers.get('content-type')}"
            )
            lines = [l for l in r.text.split("\n") if l.startswith("data:")]
            assert lines, "No SSE data lines in /chat/stream response"
            print(f"        SSE lines received: {len(lines)}")

    return len(_failures) == 0


if __name__ == "__main__":
    print(f"=== Smoke Test → {BASE_URL} ===\n")
    ok = run()
    if _failures:
        print(f"\n{len(_failures)} check(s) FAILED: {_failures}")
    else:
        print(f"\nAll checks passed.")
    sys.exit(0 if ok else 1)
