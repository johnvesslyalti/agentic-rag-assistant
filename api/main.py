import json

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Agentic RAG Assistant")


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


async def _stream_tokens(query: str):
    from agent.agent import graph
    messages = [HumanMessage(content=query)]
    async for event in graph.astream_events({"messages": messages}, version="v1"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                yield f"data: {json.dumps({'token': chunk.content})}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    return StreamingResponse(
        _stream_tokens(request.query),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}
