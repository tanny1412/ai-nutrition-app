from __future__ import annotations

import types

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


class DummyChain:
    """Lightweight chain stub returning a canned response."""

    def __init__(self, response: str = "Stub answer") -> None:
        self.response = response

    def invoke(self, _inputs):
        return types.SimpleNamespace(content=self.response)


class DummyLLM:
    """LLM stub that always returns the same content."""

    def __init__(self, response: str = "Stub answer") -> None:
        self.response = response

    def invoke(self, *_args, **_kwargs):
        return types.SimpleNamespace(content=self.response)


class StubRAGService:
    """Mocks RAG retrieval while persisting turns to memory when session_id is provided."""

    def __init__(self, memory_factory):
        self._memory_factory = memory_factory
        self.calls = []

    def answer(self, question: str, top_k: int | None = None, session_id: str | None = None, user_id: str | None = None) -> dict:
        self.calls.append({
            "question": question,
            "top_k": top_k,
            "session_id": session_id,
            "user_id": user_id,
        })

        effective_k = top_k if top_k is not None else 3
        effective_k = max(effective_k, 0)

        memory = self._memory_factory()
        if session_id and memory:
            memory.ensure_conversation(session_id, user_id=user_id)
            memory.append_message(session_id, "user", question)
            memory.append_message(session_id, "assistant", "Stub answer")

        context = [
            {"content": f"chunk-{idx}", "metadata": {"rank": idx}}
            for idx in range(effective_k)
        ]
        return {"answer": "Stub answer", "context": context}


@pytest.fixture()
def client(monkeypatch, tmp_path) -> TestClient:
    from backend.app.services.simple_agentic_pipeline import (
        SimpleAgenticPipeline,
        get_simple_agent_pipeline,
        get_rag_service,
    )
    from backend.app.services.memory_service import get_memory_service

    # Reset cached singletons so tests get a fresh configuration
    for factory in (get_simple_agent_pipeline, get_rag_service, get_memory_service):
        if hasattr(factory, "_instance"):
            delattr(factory, "_instance")

    memory_path = tmp_path / "memory.sqlite"
    monkeypatch.setenv("MEMORY_DSN", f"sqlite:///{memory_path}")
    monkeypatch.setenv("ENABLE_MEMORY", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RAG_TOP_K", "3")

    stub_rag = StubRAGService(get_memory_service)

    # Patch the simple pipeline to use stub components and avoid external calls
    from backend.app.config import get_settings

    def patched_init(self, rag_service, memory_type: str = "buffer") -> None:  # type: ignore[override]
        self.rag_service = rag_service
        self.settings = get_settings()
        self.llm = DummyLLM()
        self._memories = {}
        self.memory_type = memory_type
        self.prompt = None
        self.chain = DummyChain()

    monkeypatch.setattr(SimpleAgenticPipeline, "__init__", patched_init)

    monkeypatch.setattr(
        "backend.app.services.simple_agentic_pipeline.get_rag_service",
        lambda: stub_rag,
    )
    monkeypatch.setattr(
        "backend.app.services.rag_service.get_rag_service",
        lambda: stub_rag,
    )

    if hasattr(get_memory_service, "_instance"):
        delattr(get_memory_service, "_instance")

    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        test_client.close()
        for factory in (get_simple_agent_pipeline, get_rag_service, get_memory_service):
            if hasattr(factory, "_instance"):
                delattr(factory, "_instance")


def test_session_history_persists_turns(client: TestClient) -> None:
    create_resp = client.post("/api/sessions/create", json={"user_id": "user-123"})
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    ask_payload = {
        "user_id": "user-123",
        "session_id": session_id,
        "question": "What are good sources of protein?",
        "top_k": 2,
    }
    ask_resp = client.post("/api/ask", json=ask_payload)
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert ask_data["answer"] == "Stub answer"
    assert len(ask_data["context"]) == 2

    history_resp = client.get(f"/api/sessions/{session_id}/history")
    assert history_resp.status_code == 200
    history = history_resp.json()
    assert history["messages"]
    assert len(history["messages"]) == 2
    assert history["messages"][0]["role"] == "user"
    assert history["messages"][0]["content"] == ask_payload["question"]
    assert history["messages"][1]["role"] == "assistant"
    assert history["messages"][1]["content"] == "Stub answer"


def test_top_k_controls_context_size(client: TestClient) -> None:
    base_payload = {
        "user_id": "user-topk",
        "question": "What foods are high in protein?",
    }

    one_resp = client.post("/api/ask", json={**base_payload, "top_k": 1})
    assert one_resp.status_code == 200
    data_one = one_resp.json()
    assert len(data_one["context"]) == 1

    four_resp = client.post("/api/ask", json={**base_payload, "top_k": 4})
    assert four_resp.status_code == 200
    data_four = four_resp.json()
    assert len(data_four["context"]) == 4
