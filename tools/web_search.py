import os

from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the web for current information using Tavily."""
    try:
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "TAVILY_API_KEY is not set. Cannot perform web search."
        client = TavilyClient(api_key=api_key)
        results = client.search(query, max_results=3)
        items = results.get("results", [])
        if not items:
            return "No web results found."
        return "\n\n".join(
            f"{r.get('title', 'Result')}: {r.get('content', '')}" for r in items
        )
    except Exception as e:
        return f"Web search error: {e}"
