"""Streamlit chat UI for the Agentic RAG Assistant."""
import json
import os
import uuid

import requests
import streamlit as st

API_URL = st.sidebar.text_input(
    "API Base URL",
    value=os.getenv("API_URL", "http://localhost:8000"),
    help="URL where the FastAPI backend is running",
)

st.title("🤖 Agentic RAG Assistant")
st.caption("Ask anything — the agent picks between essay retrieval, web search, or direct answers.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown("---")
    st.markdown("**Session ID**")
    st.code(st.session_state.session_id, language=None)
    if st.button("New session"):
        try:
            requests.delete(
                f"{API_URL}/sessions/{st.session_state.session_id}/history",
                timeout=5,
            )
        except Exception:
            pass
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about startups, writing, or anything else…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        try:
            with requests.post(
                f"{API_URL}/chat/stream",
                json={"query": prompt, "session_id": st.session_state.session_id},
                stream=True,
                timeout=120,
            ) as resp:
                resp.raise_for_status()
                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    payload = line[len("data: "):]
                    if payload == "[DONE]":
                        break
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if "error" in data:
                        full_response += f"\n\n⚠️ {data['error']}"
                        break
                    token = data.get("token", "")
                    full_response += token
                    placeholder.markdown(full_response + "▌")

        except requests.exceptions.ConnectionError:
            full_response = "⚠️ Cannot reach the API. Make sure the FastAPI server is running."
        except Exception as exc:
            full_response = f"⚠️ Error: {exc}"

        placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
