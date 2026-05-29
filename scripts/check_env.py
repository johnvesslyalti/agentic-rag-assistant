#!/usr/bin/env python3
"""
Environment validator for the Agentic RAG Assistant.
Run before deploying or starting the server: python scripts/check_env.py
Exit code 0 = all required vars present; 1 = missing required items.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

_PASS = "[OK]"
_FAIL = "[FAIL]"
_WARN = "[WARN]"

_errors: list[str] = []
_warnings: list[str] = []


def _check(label: str, ok: bool, required: bool, hint: str = "") -> None:
    if ok:
        print(f"  {_PASS}  {label}")
    elif required:
        print(f"  {_FAIL}  {label}" + (f"\n         {hint}" if hint else ""))
        _errors.append(label)
    else:
        print(f"  {_WARN}  {label}" + (f"\n         {hint}" if hint else ""))
        _warnings.append(label)


def main() -> None:
    print("=== Agentic RAG Assistant — Environment Check ===\n")

    openai_key = os.getenv("OPENAI_API_KEY", "")
    _check(
        "OPENAI_API_KEY",
        ok=bool(openai_key) and openai_key != "your_openai_key_here",
        required=True,
        hint="Get a key at https://platform.openai.com/api-keys and add it to .env",
    )

    tavily_key = os.getenv("TAVILY_API_KEY", "")
    _check(
        "TAVILY_API_KEY",
        ok=bool(tavily_key) and tavily_key != "your_tavily_key_here",
        required=False,
        hint="Get a key at https://tavily.com — web_search tool will fail without it",
    )

    index_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "retriever", "faiss_index")
    )
    _check(
        "retriever/faiss_index/",
        ok=os.path.exists(index_path),
        required=False,
        hint="Run: python scripts/ingest.py  (requires OPENAI_API_KEY)",
    )

    print()

    if _errors:
        print(f"Required items missing: {', '.join(_errors)}")
        print("Copy .env.example → .env and fill in your API keys.")
        sys.exit(1)

    if _warnings:
        print(f"Optional items missing: {', '.join(_warnings)} — some features will be degraded.")
    else:
        print("All checks passed. Ready to start.")

    sys.exit(0)


if __name__ == "__main__":
    main()
