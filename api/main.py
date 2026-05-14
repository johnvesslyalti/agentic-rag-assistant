import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

from agent.agent import get_agent

app = FastAPI(title="Agentic RAG Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream tokens from the agent as server-sent events."""
    agent = get_agent()
    config = {"configurable": {"thread_id": request.session_id}}
    input_state = {"messages": [HumanMessage(content=request.query)]}

    async def generate():
        try:
            async for event in agent.astream_events(input_state, config=config, version="v2"):
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield f"data: {json.dumps({'token': chunk.content})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Non-streaming chat endpoint; returns the full response once complete."""
    agent = get_agent()
    config = {"configurable": {"thread_id": request.session_id}}
    input_state = {"messages": [HumanMessage(content=request.query)]}
    try:
        result = await agent.ainvoke(input_state, config=config)
        last_message = result["messages"][-1]
        return ChatResponse(response=last_message.content, session_id=request.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
