import os
from typing import Any, Dict

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BACKEND_URL = os.getenv("BACKEND_API_BASE", "http://localhost:8000/api")

st.set_page_config(
    page_title="AI Nutrition Coach",
    page_icon="🥗",
    layout="wide",
)

st.title("🥗 AI Nutrition Coach")

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL


def call_backend(method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
    url = f"{st.session_state.backend_url.rstrip('/')}" + path
    response = requests.request(method, url, timeout=30, **kwargs)
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text}")
    return response.json()


with st.sidebar:
    st.header("Settings")
    st.text_input(
        "Backend URL",
        help="FastAPI base path. Default assumes `uvicorn app.main:app --reload` on port 8000.",
        key="backend_url",
    )

    if st.button("Ping backend", key="ping_backend_btn"):
        try:
            health = call_backend("GET", "/health")
            st.success(f"Backend reachable: {health}")
        except Exception as exc:
            st.error(f"Unable to reach backend: {exc}")

    st.divider()
    st.subheader("Upload knowledge")
    uploaded = st.file_uploader("Add nutrition notes (.txt or .md)", type=["txt", "md", "mdx"], key="uploader")
    if uploaded and st.button("Ingest document", key="ingest_btn"):
        try:
            text = uploaded.read().decode("utf-8")
            payload = {"content": text, "title": uploaded.name}
            result = call_backend("POST", "/ingest", json=payload)
            st.success(f"Indexed {result['chunks_indexed']} chunks from {uploaded.name}")
        except Exception as exc:
            st.error(f"Ingestion failed: {exc}")


st.markdown(
    "Ask questions about your nutrition references. The app retrieves relevant context and crafts an answer.",
)

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
                response = call_backend(
                    "POST",
                    "/ask",
                    json={"question": question, "top_k": top_k},
                )
            except Exception as exc:
                st.error(f"Request failed: {exc}")
            else:
                st.markdown("### Answer")
                st.write(response["answer"])

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
