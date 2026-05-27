"""
Build the FAISS vector index from Paul Graham essays.

Usage:
    python scripts/ingest.py

Requires OPENAI_API_KEY in .env.
The index is saved to retriever/faiss_index/.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

ESSAYS = [
    ("How to Get Startup Ideas", "http://www.paulgraham.com/startupideas.html"),
    ("Do Things that Don't Scale", "http://www.paulgraham.com/ds.html"),
    ("Default Alive or Default Dead?", "http://www.paulgraham.com/aord.html"),
    ("How to Make Wealth", "http://www.paulgraham.com/wealth.html"),
    ("Keep Your Identity Small", "http://www.paulgraham.com/identity.html"),
    ("What I Worked On", "http://www.paulgraham.com/worked.html"),
    ("Writing and Speaking", "http://www.paulgraham.com/speak.html"),
    ("Hackers and Painters", "http://www.paulgraham.com/hp.html"),
    ("The Age of the Essay", "http://www.paulgraham.com/essay.html"),
    ("What Startups Are Really Like", "http://www.paulgraham.com/really.html"),
    ("Beating the Averages", "http://www.paulgraham.com/avg.html"),
    ("Startup = Growth", "http://www.paulgraham.com/growth.html"),
    ("The Hardest Lessons for Startups to Learn", "http://www.paulgraham.com/startuplessons.html"),
    ("Founder Mode", "http://www.paulgraham.com/foundermode.html"),
]

_HTML_ENTITY_MAP = {
    "&nbsp;": " ", "&amp;": "&", "&lt;": "<", "&gt;": ">",
    "&quot;": '"', "&#39;": "'", "&mdash;": "—", "&ldquo;": '"',
    "&rdquo;": '"', "&lsquo;": "'", "&rsquo;": "'",
}


def _strip_html(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    for entity, replacement in _HTML_ENTITY_MAP.items():
        text = text.replace(entity, replacement)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_essay(title: str, url: str) -> Document | None:
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        text = _strip_html(resp.text)
        if len(text) < 200:
            print(f"  Warning: very short content for '{title}' ({len(text)} chars) — skipping")
            return None
        print(f"  OK: '{title}' ({len(text):,} chars)")
        return Document(page_content=text, metadata={"title": title, "source": url})
    except Exception as exc:
        print(f"  Error fetching '{title}': {exc}")
        return None


def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set. Add it to .env and retry.")
        sys.exit(1)

    print(f"Fetching {len(ESSAYS)} Paul Graham essays...\n")
    docs = [doc for title, url in ESSAYS if (doc := _fetch_essay(title, url))]

    if not docs:
        print("\nNo essays fetched. Check your network connection.")
        sys.exit(1)

    print(f"\nFetched {len(docs)}/{len(ESSAYS)} essays. Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    print("\nBuilding FAISS index (calling OpenAI embeddings API)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_dir = os.path.join(os.path.dirname(__file__), "..", "retriever", "faiss_index")
    index_dir = os.path.abspath(index_dir)
    os.makedirs(index_dir, exist_ok=True)
    vectorstore.save_local(index_dir)

    print(f"\nFAISS index saved to {index_dir}")
    print("Next: python retriever/retriever.py   — to verify retrieval works.")


if __name__ == "__main__":
    main()
