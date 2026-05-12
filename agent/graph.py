from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.state import AgentState
from tools import ALL_TOOLS

load_dotenv()

_llm_with_tools = None
_app = None


def _get_llm():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(ALL_TOOLS)
    return _llm_with_tools


def agent_node(state: AgentState) -> dict:
    messages = state["messages"]
    # Cap to last 10 messages to avoid context window blowout (Phase 4 memory)
    if len(messages) > 10:
        messages = messages[-10:]
    response = _get_llm().invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", END: END}
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


def get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app
