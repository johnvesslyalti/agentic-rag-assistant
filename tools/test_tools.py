"""
Isolation tests for Phase 2 tools and Phase 3 agent.
Sets dummy env vars so imports work without real API keys.
Does NOT make live API or FAISS calls.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("OPENAI_API_KEY", "sk-dummy-key-for-import-test")
os.environ.setdefault("TAVILY_API_KEY", "tvly-dummy-key-for-import-test")


def test_tool_imports():
    from langchain_core.tools import BaseTool
    from tools.tools import retrieve_essays, web_search, answer_directly
    assert isinstance(retrieve_essays, BaseTool)
    assert isinstance(web_search, BaseTool)
    assert isinstance(answer_directly, BaseTool)
    print("PASS  tool imports")


def test_tool_names():
    from tools.tools import retrieve_essays, web_search, answer_directly
    assert retrieve_essays.name == "retrieve_essays"
    assert web_search.name == "web_search"
    assert answer_directly.name == "answer_directly"
    print("PASS  tool names")


def test_tool_descriptions():
    from tools.tools import retrieve_essays, web_search, answer_directly
    for t in (retrieve_essays, web_search, answer_directly):
        assert t.description, f"{t.name} has no description"
    print("PASS  tool descriptions")


def test_agent_graph_import():
    from agent.agent import graph, AgentState, TOOLS
    assert graph is not None
    assert len(TOOLS) == 3
    print("PASS  agent graph import")


def test_api_app_import():
    from api.main import app
    routes = [r.path for r in app.routes]
    assert "/chat" in routes
    assert "/health" in routes
    print("PASS  FastAPI app import and routes")


def test_retriever_missing_index():
    from retriever.retriever import retrieve
    try:
        retrieve("test query")
        print("PASS  retriever (index found)")
    except FileNotFoundError as e:
        print(f"PASS  retriever (no index, raises FileNotFoundError as expected: {e})")


if __name__ == "__main__":
    tests = [
        test_tool_imports,
        test_tool_names,
        test_tool_descriptions,
        test_agent_graph_import,
        test_api_app_import,
        test_retriever_missing_index,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
            failures.append(t.__name__)

    print()
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    else:
        print("All tests passed.")
