"""
Ingest script tests — no network calls, no API keys required.
Tests _strip_html, _fetch_essay, and main() validation.
Run: pytest tests/test_ingest.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import scripts.ingest as ingest_mod


# ---------------------------------------------------------------------------
# _strip_html
# ---------------------------------------------------------------------------

def test_strip_html_removes_tags():
    result = ingest_mod._strip_html("<p>Hello <b>world</b></p>")
    assert "Hello" in result
    assert "world" in result
    assert "<" not in result
    assert ">" not in result


def test_strip_html_decodes_common_entities():
    result = ingest_mod._strip_html("AT&amp;T &mdash; &ldquo;smart&rdquo; &#39;quotes&#39;")
    assert "AT&T" in result
    assert "—" in result
    assert '"' in result
    assert "'" in result
    assert "&amp;" not in result


def test_strip_html_collapses_whitespace():
    result = ingest_mod._strip_html("<p>too   many   spaces</p>")
    assert "  " not in result


def test_strip_html_empty_input():
    assert ingest_mod._strip_html("") == ""


def test_strip_html_plain_text_unchanged():
    text = "No HTML here."
    assert ingest_mod._strip_html(text) == text


def test_strip_html_nested_tags():
    html = "<div><h1>Title</h1><p>Body <em>text</em></p></div>"
    result = ingest_mod._strip_html(html)
    assert "Title" in result
    assert "Body" in result
    assert "text" in result
    assert "<" not in result


# ---------------------------------------------------------------------------
# _fetch_essay
# ---------------------------------------------------------------------------

def _mock_ok_response(body: str) -> MagicMock:
    r = MagicMock()
    r.text = body
    r.raise_for_status = MagicMock()
    return r


def _mock_error_response() -> MagicMock:
    r = MagicMock()
    r.raise_for_status = MagicMock(side_effect=Exception("HTTP 404"))
    return r


LONG_BODY = "<p>" + "Paul Graham says things about startups. " * 10 + "</p>"


def test_fetch_essay_returns_document_on_success():
    with patch("requests.get", return_value=_mock_ok_response(LONG_BODY)):
        doc = ingest_mod._fetch_essay("Test Essay", "http://example.com/essay")
    assert doc is not None
    assert "Paul Graham" in doc.page_content
    assert doc.metadata["title"] == "Test Essay"
    assert doc.metadata["source"] == "http://example.com/essay"


def test_fetch_essay_returns_none_on_short_content():
    short_html = "<p>Too short.</p>"
    with patch("requests.get", return_value=_mock_ok_response(short_html)):
        doc = ingest_mod._fetch_essay("Short", "http://example.com/short")
    assert doc is None


def test_fetch_essay_returns_none_on_network_error():
    with patch("requests.get", side_effect=Exception("Connection refused")):
        doc = ingest_mod._fetch_essay("Fails", "http://example.com/fail")
    assert doc is None


def test_fetch_essay_returns_none_on_http_error():
    with patch("requests.get", return_value=_mock_error_response()):
        doc = ingest_mod._fetch_essay("Gone", "http://example.com/gone")
    assert doc is None


def test_fetch_essay_document_has_plain_text():
    """Document page_content should have no HTML tags."""
    with patch("requests.get", return_value=_mock_ok_response(LONG_BODY)):
        doc = ingest_mod._fetch_essay("Clean", "http://example.com/clean")
    assert doc is not None
    assert "<" not in doc.page_content
    assert ">" not in doc.page_content


# ---------------------------------------------------------------------------
# main() — validation guard
# ---------------------------------------------------------------------------

def test_main_exits_without_openai_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        ingest_mod.main()
    assert exc_info.value.code == 1
