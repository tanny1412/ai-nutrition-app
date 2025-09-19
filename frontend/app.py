import os
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BACKEND_URL = os.getenv("BACKEND_API_BASE", "http://localhost:8000/api")


def _backend_request(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    url = f"{st.session_state.backend_url.rstrip('/')}" + path
    response = requests.request(method, url, timeout=30, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    try:
        return response.json()
    except ValueError:
        return {}


def refresh_sessions(user_id: Optional[str] = None) -> None:
    params = {"user_id": user_id} if user_id else None
    try:
        result = _backend_request("GET", "/sessions/list", params=params)
        if isinstance(result, list):
            st.session_state.sessions = result
        else:
            st.session_state.sessions = []
            st.warning("Unexpected response when listing sessions.")
    except Exception as exc:  # noqa: BLE001
        st.session_state.sessions = []
        st.warning(f"Could not refresh sessions: {exc}")


def create_session(user_id: Optional[str], title: Optional[str]) -> Optional[str]:
    payload: Dict[str, Any] = {}
    if user_id:
        payload["user_id"] = user_id
    if title:
        payload["title"] = title
    try:
        result = _backend_request("POST", "/sessions/create", json=payload)
        session_id = result.get("session_id")
        if session_id:
            refresh_sessions(user_id)
        return session_id
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to create session: {exc}")
        return None


def load_session_history(session_id: str) -> List[Dict[str, str]]:
    try:
        result = _backend_request("GET", f"/sessions/{session_id}/history")
        messages = result.get("messages", [])
        if isinstance(messages, list):
            return messages
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Could not fetch history: {exc}")
    return []


st.set_page_config(
    page_title="AI Nutrition Coach",
    page_icon="🥗",
    layout="wide",
)

st.title("🥗 AI Nutrition Coach")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL
if "sessions" not in st.session_state:
    st.session_state.sessions: List[Dict[str, Any]] = []
if "active_session" not in st.session_state:
    st.session_state.active_session: Optional[str] = None
if "active_user" not in st.session_state:
    st.session_state.active_user: Optional[str] = None
if "session_title_input" not in st.session_state:
    st.session_state.session_title_input = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history: List[Dict[str, str]] = []


with st.sidebar:
    st.header("Settings")
    st.text_input(
        "Backend URL",
        help="FastAPI base path. Default assumes `uvicorn app.main:app --reload` on port 8000.",
        key="backend_url",
    )
    st.text_input(
        "User ID (optional)",
        help="Associate sessions with a user; leave blank for shared conversations.",
        key="active_user",
    )

    if st.button("Ping backend", key="ping_backend_btn"):
        try:
            health = _backend_request("GET", "/health")
            st.success(f"Backend reachable: {health}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Unable to reach backend: {exc}")

    st.divider()
    st.subheader("Conversation")
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("New conversation", key="new_convo_btn"):
            title = st.session_state.session_title_input.strip() or None
            session_id = create_session(st.session_state.active_user, title)
            if session_id:
                st.session_state.active_session = session_id
                st.session_state.chat_history = load_session_history(session_id)
                st.success("Started a new conversation.")
    with col2:
        if st.button("Refresh list", key="refresh_sessions_btn"):
            refresh_sessions(st.session_state.active_user)

    st.text_input(
        "Conversation title (optional)",
        key="session_title_input",
        help="Used when creating a new conversation.",
    )

    if not st.session_state.sessions:
        refresh_sessions(st.session_state.active_user)

    session_options = [item.get("id") for item in st.session_state.sessions]
    labels = [
        f"{item.get('title') or item['id'][:8]} • Updated {item.get('updated_at', '')[:19]}"
        for item in st.session_state.sessions
    ]

    def _format_session(sid: str) -> str:
        for label, option in zip(labels, session_options):
            if option == sid:
                return label
        return sid

    if session_options:
        selected = st.selectbox(
            "Active conversation",
            session_options,
            index=session_options.index(st.session_state.active_session)
            if st.session_state.active_session in session_options
            else 0,
            format_func=_format_session,
        )
        if selected != st.session_state.active_session:
            st.session_state.active_session = selected
            st.session_state.chat_history = load_session_history(selected)
    else:
        st.info("No saved conversations yet. Create one above.")

    st.divider()
    st.subheader("Upload knowledge")
    uploaded = st.file_uploader("Add nutrition notes (.txt or .md)", type=["txt", "md", "mdx"], key="uploader")
    if uploaded and st.button("Ingest document", key="ingest_btn"):
        try:
            text = uploaded.read().decode("utf-8")
            payload = {"content": text, "title": uploaded.name}
            result = _backend_request("POST", "/ingest", json=payload)
            st.success(f"Indexed {result['chunks_indexed']} chunks from {uploaded.name}")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ingestion failed: {exc}")


st.markdown(
    "Ask questions about your nutrition references. The app retrieves relevant context and crafts an answer.",
)

if st.session_state.active_session:
    st.info(f"Responding within session `{st.session_state.active_session}`")

with st.form("qa-form"):
    question = st.text_area(
        "Nutrition question",
        placeholder="e.g. What should I eat before a long run?",
        height=120,
        key="question_input",
    )
    top_k = st.slider("Number of context snippets", min_value=2, max_value=8, value=4, key="topk_slider")
    submitted = st.form_submit_button("Get personalized guidance", key="submit_btn")

if submitted:
    if not question.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Consulting your nutrition knowledge base..."):
            try:
                payload = {"question": question, "top_k": top_k}
                if st.session_state.active_session:
                    payload["session_id"] = st.session_state.active_session
                if st.session_state.active_user:
                    payload["user_id"] = st.session_state.active_user
                response = _backend_request("POST", "/ask", json=payload)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")
            else:
                st.markdown("### Answer")
                st.write(response.get("answer", ""))

                if response.get("context"):
                    with st.expander("Show retrieved context"):
                        for idx, chunk in enumerate(response["context"], start=1):
                            metadata = chunk.get("metadata", {})
                            source = metadata.get("source", f"Chunk {idx}")
                            st.markdown(f"**{idx}. {source}**")
                            st.write(chunk.get("content", ""))

                            meta_lines = []
                            if metadata.get("section"):
                                meta_lines.append(f"Section: {metadata['section']}")
                            if metadata.get("page"):
                                meta_lines.append(f"Page: {metadata['page']}")
                            if metadata.get("chunk_index") is not None and metadata.get("num_chunks"):
                                meta_lines.append(
                                    f"Chunk {metadata['chunk_index']} of {metadata['num_chunks']}",
                                )
                            if metadata.get("path"):
                                meta_lines.append(str(metadata["path"]))
                            if meta_lines:
                                st.caption(" | ".join(meta_lines))
                else:
                    st.info("No supporting context was returned. Try ingesting more nutrition documents.")

                if st.session_state.active_session:
                    st.session_state.chat_history = load_session_history(st.session_state.active_session)

if st.session_state.chat_history:
    st.markdown("### Conversation history")
    for message in st.session_state.chat_history[-10:]:  # show last 10 turns
        role = message.get("role", "user").capitalize()
        content = message.get("content", "")
        st.markdown(f"**{role}:** {content}")
