import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

_vectorstore = None

def _load_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        if os.path.exists(index_path):
            _vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            raise FileNotFoundError(
                "FAISS index not found. Place your faiss_index/ folder inside retriever/"
            )
    return _vectorstore


def retrieve(query: str, k: int = 5) -> list[str]:
    """
    Retrieve top-k relevant chunks for a given query.
    Returns a list of text strings.
    """
    store = _load_vectorstore()
    docs = store.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


if __name__ == "__main__":
    results = retrieve("What does Paul Graham say about startups?")
    for i, r in enumerate(results, 1):
        print(f"--- Chunk {i} ---")
        print(r)
        print()
