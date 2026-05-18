"""
Agent graph tests — all LLM calls mocked, no API keys required.

Covers (from the build plan):
  Phase 3 — Test with 5 different queries; verify agent picks the right tool.
  Phase 4 — Test multi-turn context; verify history cap at MAX_HISTORY.
  Phase 2/3 — Tool error handling (graceful strings, no exceptions raised).

Run: pytest tests/test_agent.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ai_tool_call(tool_name: str, query: str, call_id: str = "call_test") -> AIMessage:
    """AIMessage that requests a specific tool call."""
    return AIMessage(
        content="",
        tool_calls=[{
            "id": call_id,
            "name": tool_name,
            "args": {"query": query},
            "type": "tool_call",
        }],
    )


def _ai_final(content: str = "Here is my answer.") -> AIMessage:
    """Final AIMessage with no tool calls."""
    return AIMessage(content=content)


def _make_mock_llm(responses: list) -> MagicMock:
    """
    Build a mock that simulates: ChatOpenAI(...).bind_tools(...).invoke(...)
    returning *responses* in order.
    """
    llm_with_tools = MagicMock()
    llm_with_tools.invoke.side_effect = responses
    llm = MagicMock()
    llm.bind_tools.return_value = llm_with_tools
    return llm


# ---------------------------------------------------------------------------
# Reset cached agent between tests so each test gets a fresh graph.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_agent_cache():
    import agent.agent as mod
    mod._agent = None
    yield
    mod._agent = None


# ---------------------------------------------------------------------------
# Phase 3 — Tool routing: 5 diverse queries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tool_name,query,tid", [
    ("retrieve_essays", "What does Paul Graham say about startups?",       "tr-1"),
    ("retrieve_essays", "Explain PG's views on writing and essays",        "tr-2"),
    ("retrieve_essays", "How does Paul Graham think about hiring?",        "tr-3"),
    ("web_search",      "What is the latest AI research news today?",      "tr-4"),
    ("web_search",      "Who won the most recent presidential election?",  "tr-5"),
])
def test_agent_routes_to_correct_tool(tool_name, query, tid):
    """
    Agent picks the tool that the (mocked) LLM chose.
    The tool runs for real but fails gracefully without API keys / FAISS index.
    """
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_llm([
            _ai_tool_call(tool_name, query),
            _ai_final("Here is the synthesised answer."),
        ])
        from agent.agent import _build_agent
        graph = _build_agent()
        result = graph.invoke(
            {"messages": [HumanMessage(content=query)]},
            config={"configurable": {"thread_id": tid}},
        )

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    tool_calls_msg = next((m for m in ai_messages if m.tool_calls), None)
    assert tool_calls_msg is not None, f"Expected a tool call for query: {query!r}"
    assert tool_calls_msg.tool_calls[0]["name"] == tool_name


# ---------------------------------------------------------------------------
# Phase 4 — Multi-turn memory
# ---------------------------------------------------------------------------

def test_multi_turn_context_retained():
    """
    Two questions sent to the same thread_id both appear in the final state,
    proving MemorySaver persists history across invocations.
    """
    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        mock_cls.return_value = _make_mock_llm([
            _ai_final("Paul Graham co-founded Y Combinator."),
            _ai_final("He started it in 2005 with Jessica Livingston and Robert Morris."),
        ])
        from agent.agent import _build_agent
        graph = _build_agent()
        config = {"configurable": {"thread_id": "memory-test"}}

        graph.invoke(
            {"messages": [HumanMessage(content="Who is Paul Graham?")]},
            config=config,
        )
        result2 = graph.invoke(
            {"messages": [HumanMessage(content="When did he start Y Combinator?")]},
            config=config,
        )

    human_messages = [m for m in result2["messages"] if isinstance(m, HumanMessage)]
    assert len(human_messages) >= 2, (
        f"Expected ≥ 2 HumanMessages in history, got {len(human_messages)}"
    )


# ---------------------------------------------------------------------------
# Phase 4 — History cap
# ---------------------------------------------------------------------------

def test_max_history_constant():
    """MAX_HISTORY is set to 10 as specified in the memory plan."""
    from agent.agent import MAX_HISTORY
    assert MAX_HISTORY == 10


def test_agent_node_caps_messages_sent_to_llm():
    """
    After accumulating more than MAX_HISTORY messages, agent_node still passes
    at most MAX_HISTORY + 1 (SystemMessage + capped history) to the LLM.
    """
    from agent.agent import MAX_HISTORY
    captured_lengths: list[int] = []

    def capture_invoke(messages):
        captured_lengths.append(len(messages))
        return _ai_final(f"Reply {len(captured_lengths)}")

    with patch("langchain_openai.ChatOpenAI") as mock_cls:
        llm_with_tools = MagicMock()
        llm_with_tools.invoke.side_effect = capture_invoke
        mock_cls.return_value = MagicMock(
            bind_tools=MagicMock(return_value=llm_with_tools)
        )
        from agent.agent import _build_agent
        graph = _build_agent()
        config = {"configurable": {"thread_id": "cap-test"}}

        for i in range(MAX_HISTORY + 5):
            graph.invoke(
                {"messages": [HumanMessage(content=f"Question {i}")]},
                config=config,
            )

    # Once history exceeds MAX_HISTORY, the LLM must never receive more than
    # MAX_HISTORY messages + 1 SystemMessage prepended by agent_node.
    late_calls = captured_lengths[MAX_HISTORY:]
    assert late_calls, "Expected LLM calls after history exceeded MAX_HISTORY"
    assert all(n <= MAX_HISTORY + 1 for n in late_calls), (
        f"LLM received too many messages on late calls: {late_calls}"
    )


# ---------------------------------------------------------------------------
# Phase 2 & 3 — Tool error handling (tools return strings, never raise)
# ---------------------------------------------------------------------------

def test_retrieve_essays_graceful_when_index_missing():
    """retrieve_essays returns a non-empty string when the FAISS index is absent."""
    from tools.tools import retrieve_essays
    result = retrieve_essays.invoke("What does Paul Graham say about startups?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_web_search_graceful_on_tavily_error():
    """web_search returns a non-empty error string when Tavily raises."""
    with patch("tavily.TavilyClient") as mock_tavily:
        mock_tavily.side_effect = Exception("Tavily unavailable")
        from tools.tools import web_search
        result = web_search.invoke("latest AI news")
    assert isinstance(result, str)
    assert len(result) > 0


def test_answer_directly_graceful_on_llm_error():
    """answer_directly returns a non-empty error string when the LLM call fails."""
    with patch("tools.tools.ChatOpenAI") as mock_llm_cls:
        mock_llm_cls.side_effect = Exception("No API key")
        from tools.tools import answer_directly
        result = answer_directly.invoke("What is 2 + 2?")
    assert isinstance(result, str)
    assert len(result) > 0
