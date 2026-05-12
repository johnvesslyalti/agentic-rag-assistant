import operator
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from tools.tools import answer_directly, retrieve_essays, web_search

TOOLS = [retrieve_essays, web_search, answer_directly]
MAX_HISTORY = 10


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


_llm_with_tools = None


def _get_llm():
    global _llm_with_tools
    if _llm_with_tools is None:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        _llm_with_tools = llm.bind_tools(TOOLS)
    return _llm_with_tools


def _agent_node(state: AgentState) -> dict:
    history = list(state["messages"])[-MAX_HISTORY:]
    response = _get_llm().invoke(history)
    return {"messages": [response]}


def _should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return END


_tool_node = ToolNode(TOOLS)

_builder = StateGraph(AgentState)
_builder.add_node("agent", _agent_node)
_builder.add_node("tools", _tool_node)
_builder.set_entry_point("agent")
_builder.add_conditional_edges(
    "agent",
    _should_continue,
    {"tools": "tools", END: END},
)
_builder.add_edge("tools", "agent")

graph = _builder.compile()
