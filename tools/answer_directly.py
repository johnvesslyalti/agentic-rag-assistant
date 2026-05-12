import os

from langchain_core.tools import tool


@tool
def answer_directly(query: str) -> str:
    """Answer a factual or conversational question directly from the language model's training data, without retrieval or web search."""
    try:
        from langchain_openai import ChatOpenAI

        if not os.getenv("OPENAI_API_KEY"):
            return "OPENAI_API_KEY is not set. Cannot answer directly."
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        response = llm.invoke(query)
        return response.content
    except Exception as e:
        return f"Direct answer error: {e}"
