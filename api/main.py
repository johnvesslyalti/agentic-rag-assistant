import json

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Agentic RAG Assistant", version="1.0.0")

# In-memory session store: session_id -> list[BaseMessage]
_sessions: dict = {}


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


def _get_agent():
    from agent.graph import get_app

    return get_app()


def _get_history(session_id: str) -> list:
    return list(_sessions.get(session_id, []))


def _append_history(session_id: str, new_messages: list):
    history = _sessions.get(session_id, [])
    history = (history + new_messages)[-10:]  # cap to last 10
    _sessions[session_id] = history


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    history = _get_history(request.session_id)
    human_msg = HumanMessage(content=request.query)
    state = {"messages": history + [human_msg]}

    result = await _get_agent().ainvoke(state)
    ai_message = result["messages"][-1]

    _append_history(request.session_id, [human_msg, ai_message])
    return ChatResponse(response=ai_message.content, session_id=request.session_id)


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    history = _get_history(request.session_id)
    human_msg = HumanMessage(content=request.query)
    state = {"messages": history + [human_msg]}

    async def generate():
        final_ai_msg = None
        async for chunk in _get_agent().astream(state):
            for node_name, node_output in chunk.items():
                if node_name == "agent" and "messages" in node_output:
                    for msg in node_output["messages"]:
                        if getattr(msg, "content", None):
                            final_ai_msg = msg
                            yield f"data: {json.dumps({'token': msg.content, 'node': node_name})}\n\n"
        if final_ai_msg is not None:
            _append_history(request.session_id, [human_msg, final_ai_msg])
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok"}
