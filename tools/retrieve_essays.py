from langchain_core.tools import tool
from retriever.retriever import retrieve


@tool
def retrieve_essays(query: str) -> str:
    """Search the essay knowledge base for relevant content matching the query."""
    try:
        chunks = retrieve(query, k=5)
        if not chunks:
            return "No relevant content found in the essay index."
        return "\n\n---\n\n".join(chunks)
    except FileNotFoundError:
        return "Essay index not available. Place a faiss_index/ folder inside retriever/."
    except Exception as e:
        return f"Retrieval error: {e}"
