import os

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


@tool
def retrieve_essays(query: str) -> str:
    """Search the local essay knowledge base for relevant content about startups, writing, and technology."""
    from retriever.retriever import retrieve
    try:
        chunks = retrieve(query, k=5)
        if not chunks:
            return "No relevant content found in the knowledge base."
        return "\n\n---\n\n".join(chunks)
    except FileNotFoundError:
        return "Knowledge base index not found. Please add a faiss_index/ folder to retriever/."


@tool
def web_search(query: str) -> str:
    """Search the web for current, up-to-date information using Tavily."""
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(query, max_results=5)
    if not results.get("results"):
        return "No web results found."
    parts = []
    for r in results["results"]:
        parts.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r.get('url', '')}\n"
            f"Content: {r.get('content', '')}"
        )
    return "\n\n---\n\n".join(parts)


@tool
def answer_directly(query: str) -> str:
    """Answer the query using the LLM's built-in knowledge without external search. Use for general questions."""
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return llm.invoke(query).content
