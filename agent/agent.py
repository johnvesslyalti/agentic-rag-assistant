import operator
from typing import TypedDict, Annotated, List

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()

try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    from langgraph_checkpoint.memory import MemorySaver

from tools.tools import retrieve_essays, web_search, answer_directly

SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to Paul Graham's essays and the web.\n"
    "When answering:\n"
    "- Use `retrieve_essays` for questions about startups, writing, or technology from PG's essays.\n"
    "- Use `web_search` for current events or recent information not in the essays.\n"
    "- Use `answer_directly` only for simple factual questions you can answer confidently.\n"
    "Always reason about which tool fits best before calling it."
)

MAX_HISTORY = 10


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]


_TOOLS = [retrieve_essays, web_search, answer_directly]
_agent = None


def _build_agent():
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    llm_with_tools = llm.bind_tools(_TOOLS)
    tool_node = ToolNode(_TOOLS)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        # Send only the last MAX_HISTORY messages to avoid context window blowout
        context = messages[-MAX_HISTORY:] if len(messages) > MAX_HISTORY else messages
        if not any(isinstance(m, SystemMessage) for m in context):
            context = [SystemMessage(content=SYSTEM_PROMPT)] + context
        response = llm_with_tools.invoke(context)
        return {"messages": [response]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver())


def get_agent():
    global _agent
    if _agent is None:
        _agent = _build_agent()
    return _agent
