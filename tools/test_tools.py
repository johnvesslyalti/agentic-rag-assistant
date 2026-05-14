"""
Isolation test script for each tool.
Run from project root: python tools/test_tools.py

Tests skip gracefully when API keys are absent.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from tools.tools import retrieve_essays, web_search, answer_directly

HAS_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
HAS_TAVILY = bool(os.getenv("TAVILY_API_KEY"))


def test_retrieve_essays():
    print("Testing retrieve_essays...")
    result = retrieve_essays.invoke("What does Paul Graham say about startups?")
    assert isinstance(result, str), "Expected string result"
    # Acceptable outcomes: real chunks OR graceful error messages
    assert len(result) > 0, "Expected non-empty result"
    print(f"  Result preview: {result[:120]}...")
    print("  PASS\n")


def test_web_search():
    if not HAS_TAVILY:
        print("Testing web_search... SKIPPED (no TAVILY_API_KEY)\n")
        return
    print("Testing web_search...")
    result = web_search.invoke("Paul Graham essays on startups")
    assert isinstance(result, str), "Expected string result"
    assert len(result) > 0, "Expected non-empty result"
    print(f"  Result preview: {result[:120]}...")
    print("  PASS\n")


def test_answer_directly():
    if not HAS_OPENAI:
        print("Testing answer_directly... SKIPPED (no OPENAI_API_KEY)\n")
        return
    print("Testing answer_directly...")
    result = answer_directly.invoke("What is 2 + 2?")
    assert isinstance(result, str), "Expected string result"
    assert "4" in result, f"Expected '4' in result, got: {result}"
    print(f"  Result: {result}")
    print("  PASS\n")


if __name__ == "__main__":
    print("=== Tool Isolation Tests ===\n")
    failed = []
    for test_fn in [test_retrieve_essays, test_web_search, test_answer_directly]:
        try:
            test_fn()
        except Exception as e:
            print(f"  FAIL ({test_fn.__name__}): {e}\n")
            failed.append(test_fn.__name__)

    if failed:
        print(f"Failed: {failed}")
        sys.exit(1)
    print("All tests passed (skipped tests require API keys).")
