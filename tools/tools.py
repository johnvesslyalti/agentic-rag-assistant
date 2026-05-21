import os
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


@tool
def retrieve_essays(query: str) -> str:
    """Search Paul Graham's essays for relevant information about startups, writing, and technology."""
    try:
        from retriever.retriever import retrieve_with_sources
        results = retrieve_with_sources(query, k=5)
        if not results:
            return "No relevant content found in essays."
        return "\n\n---\n\n".join(
            f"[Source: {r['title']}]\n{r['content']}"
            for r in results
        )
    except FileNotFoundError:
        return "Essay retrieval unavailable: FAISS index not found. Place faiss_index/ inside retriever/."
    except Exception as e:
        return f"Retrieval error: {str(e)}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information, news, or facts not covered in Paul Graham's essays."""
    try:
        from tavily import TavilyClient
        client = TavilyClient()
        response = client.search(query, max_results=3)
        results = response.get("results", [])
        if not results:
            return "No web results found."
        return "\n\n---\n\n".join(
            f"**{r.get('title', 'Result')}**\n{r.get('content', '')}"
            for r in results
        )
    except Exception as e:
        return f"Web search error: {str(e)}"


@tool
def answer_directly(query: str) -> str:
    """Answer the question directly from LLM knowledge when no retrieval is needed."""
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        response = llm.invoke(query)
        return response.content
    except Exception as e:
        return f"Direct answer error: {str(e)}"
