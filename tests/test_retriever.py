"""
Retriever module tests — uses mock FAISS, no API keys required.
Run: pytest tests/test_retriever.py -v
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_vectorstore():
    """Clear the module-level cache between tests."""
    import retriever.retriever as mod
    mod._vectorstore = None


def _make_mock_doc(content: str) -> MagicMock:
    doc = MagicMock()
    doc.page_content = content
    return doc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_retrieve_raises_when_index_missing():
    _reset_vectorstore()
    import retriever.retriever as mod

    with patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError, match="FAISS index not found"):
            mod._load_vectorstore()


def test_retrieve_returns_strings_from_mock_store():
    import retriever.retriever as mod

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = [
        _make_mock_doc("Paul Graham says do things that don't scale."),
        _make_mock_doc("Great startups often look like bad ideas at first."),
    ]
    mod._vectorstore = mock_store

    results = mod.retrieve("startups", k=2)

    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(r, str) for r in results)
    assert "don't scale" in results[0]
    mock_store.similarity_search.assert_called_once_with("startups", k=2)

    _reset_vectorstore()


def test_retrieve_returns_empty_list_on_no_results():
    import retriever.retriever as mod

    mock_store = MagicMock()
    mock_store.similarity_search.return_value = []
    mod._vectorstore = mock_store

    results = mod.retrieve("xyzzy obscure query", k=5)
    assert results == []

    _reset_vectorstore()


def test_vectorstore_is_cached():
    """Second call to _load_vectorstore() should reuse the cached instance."""
    import retriever.retriever as mod

    mock_store = MagicMock()
    mod._vectorstore = mock_store  # pre-populate cache

    result = mod._load_vectorstore()
    assert result is mock_store

    _reset_vectorstore()
