"""Quick test — call each tool in isolation and verify output."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools import answer_directly, retrieve_essays, web_search


def test_retrieve_essays():
    print("Testing retrieve_essays...")
    result = retrieve_essays.invoke({"query": "What does Paul Graham say about startups?"})
    print(f"  result (first 200 chars): {result[:200]}")
    assert isinstance(result, str) and len(result) > 0
    print("  PASS\n")


def test_web_search():
    print("Testing web_search...")
    result = web_search.invoke({"query": "latest AI news 2025"})
    print(f"  result (first 200 chars): {result[:200]}")
    assert isinstance(result, str) and len(result) > 0
    print("  PASS\n")


def test_answer_directly():
    print("Testing answer_directly...")
    result = answer_directly.invoke({"query": "What is the capital of France?"})
    print(f"  result: {result[:200]}")
    assert isinstance(result, str) and len(result) > 0
    print("  PASS\n")


if __name__ == "__main__":
    test_retrieve_essays()  # gracefully handles missing index

    if os.getenv("TAVILY_API_KEY"):
        test_web_search()
    else:
        print("Skipping web_search test — TAVILY_API_KEY not set\n")

    if os.getenv("OPENAI_API_KEY"):
        test_answer_directly()
    else:
        print("Skipping answer_directly test — OPENAI_API_KEY not set\n")

    print("All available tests passed!")
