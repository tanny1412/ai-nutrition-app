import sys
import types
from typing import Dict

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.services.agentic_pipeline import (
    AgentDependencies,
    AgenticPipeline,
    InMemoryKeyValueStore,
    build_agent_graph,
    classify_intent,
)


class DummyFAISS:
    def __init__(self, documents=None, embeddings=None) -> None:
        self.documents = list(documents or [])
        self.embeddings = embeddings
        self.docstore = types.SimpleNamespace(_dict={})

    @classmethod
    def load_local(cls, *args, **kwargs):
        return cls()

    @classmethod
    def from_documents(cls, documents, embeddings):
        return cls(documents, embeddings)

    def add_documents(self, documents):
        self.documents.extend(documents)
        for doc in documents:
            key = getattr(doc, "metadata", {}).get("id", len(self.docstore._dict))
            self.docstore._dict[key] = doc

    def similarity_search_with_score(self, query, k=4):  # noqa: ARG002
        return []

    def save_local(self, *args, **kwargs):
        return None

    def similarity_search(self, query, k=4):  # noqa: ARG002
        return []


sys.modules.setdefault(
    "langchain_community.vectorstores",
    types.SimpleNamespace(FAISS=DummyFAISS),
)


class DummyRAGService:
    def __init__(self) -> None:
        self.calls = []

    def answer(self, question: str, top_k: int | None = None) -> dict:  # noqa: ARG002
        self.calls.append(question)
        return {
            "answer": f"RAG: contextual answer for '{question}'",
            "context": [
                {
                    "content": "Protein sources include beans and lentils.",
                    "metadata": {"source": "test_doc.md"},
                }
            ],
        }


class DummyLLM:
    def __init__(self, response: str = "Hello!") -> None:
        self.response = response
        self.calls = []

    def __call__(self, question: str) -> str:
        self.calls.append(question)
        
        # Mock LLM responses for memory system
        if "intent classifier" in question.lower():
            # Extract the actual message from the prompt
            if "Message:" in question:
                actual_message = question.split("Message:")[-1].strip()
            else:
                actual_message = question
            
            # Check for personal information statements
            if any(word in actual_message.lower() for word in ["my name is", "i like", "i prefer", "i am allergic", "my goal"]):
                return "memory"
            # Check for personal information questions
            elif any(word in actual_message.lower() for word in ["what's my", "what do i", "what are my", "tell me about"]):
                return "memory"
            # Check for nutrition questions
            elif any(word in actual_message.lower() for word in ["protein", "nutrition", "diet", "calories", "vitamins"]):
                return "rag"
            else:
                return "chitchat"
        
        elif "extract personal information" in question.lower():
            if "my name is" in question.lower():
                # Extract the name from the message
                if "Jamie" in question:
                    return '{"name": "Jamie"}'
                elif "Tanish" in question:
                    return '{"name": "Tanish"}'
                else:
                    return '{"name": "Unknown"}'
            elif "i like" in question.lower():
                return '{"likes": "white rice"}'
            elif "i prefer" in question.lower():
                return '{"preferences": "brown rice"}'
            elif "i love to eat fish" in question.lower():
                return '{"likes": "fish"}'
            else:
                return "{}"
        
        elif "asking about personal information" in question.lower():
            # Only return "yes" for actual questions, not statements
            # Extract the actual message from the prompt
            if "Message:" in question:
                actual_message = question.split("Message:")[-1].strip()
            else:
                actual_message = question
            
            # Look for question words at the beginning or question marks
            if (actual_message.strip().startswith(("what", "tell me", "do i", "am i", "are my")) or 
                "?" in actual_message or 
                actual_message.strip().endswith("?")):
                return "yes"
            else:
                return "no"
        
        elif "personal information:" in question.lower():
            if "name: Tanish" in question and "what's my name" in question.lower():
                return "Your name is Tanish."
            elif "name: Jamie" in question and "what's my name" in question.lower():
                return "Your name is Jamie."
            elif "likes: white rice" in question and "what do i like" in question.lower():
                return "You like white rice."
            else:
                return "I don't have that information about you yet."
        
        elif "acknowledge it naturally" in question.lower():
            return "Got it, I'll remember that!"
        
        elif "just mentioned:" in question.lower():
            # Enhanced response for RAG/chitchat with personal info
            if "fish" in question.lower():
                return "Great! Fish is an excellent source of protein and omega-3 fatty acids. I'll remember that you love fish!"
            else:
                return "Thanks for sharing that with me!"
        
        return self.response


@pytest.fixture()
def deps() -> AgentDependencies:
    rag = DummyRAGService()
    store = InMemoryKeyValueStore()
    llm = DummyLLM(response="Chitchat reply")
    return AgentDependencies(rag_service=rag, memory_store=store, chat_llm=llm)


def test_classifier_routes_memory():
    # Test with a mock LLM that returns "memory"
    class MockLLM:
        def __call__(self, question: str) -> str:
            return "memory"
    
    llm = MockLLM()
    assert classify_intent("My name is Sam", llm) == "memory"
    assert classify_intent("what's my name?", llm) == "memory"


def test_classifier_routes_rag():
    # Test with a mock LLM that returns "rag"
    class MockLLM:
        def __call__(self, question: str) -> str:
            return "rag"
    
    llm = MockLLM()
    assert classify_intent("What are good sources of protein?", llm) == "rag"


def test_classifier_routes_chitchat():
    # Test with a mock LLM that returns "chitchat"
    class MockLLM:
        def __call__(self, question: str) -> str:
            return "chitchat"
    
    llm = MockLLM()
    assert classify_intent("How's the weather today?", llm) == "chitchat"


def test_memory_store_and_recall(deps: AgentDependencies):
    pipeline = AgenticPipeline(deps)
    store = deps.memory_store  # type: ignore[assignment]

    # Test name storage and recall
    response = pipeline.ask("user1", "My name is Tanish")
    assert response["route"] == "memory"
    assert "remember" in response["answer"].lower()

    response = pipeline.ask("user1", "what's my name?")
    assert response["route"] == "memory"
    assert "Tanish" in response["answer"]
    assert store.get("user1", "name") == "Tanish"

    # Test preferences storage and recall
    response = pipeline.ask("user1", "I like white rice")
    assert response["route"] == "memory"
    assert "remember" in response["answer"].lower()
    
    response = pipeline.ask("user1", "what do I like?")
    assert response["route"] == "memory"
    assert "white rice" in response["answer"].lower()
    assert store.get("user1", "likes") == "white rice"


def test_rag_node_invoked_for_nutrition_question(deps: AgentDependencies):
    pipeline = AgenticPipeline(deps)
    response = pipeline.ask("user2", "What are good sources of protein?")
    assert response["route"] == "rag"
    assert "RAG" in response["answer"]
    assert response["context"]
    assert deps.rag_service.calls  # type: ignore[attr-defined]


def test_chitchat_node_handles_smalltalk(deps: AgentDependencies):
    pipeline = AgenticPipeline(deps)
    response = pipeline.ask("user3", "Hello there!")
    assert response["route"] == "chitchat"
    assert response["answer"] == "Chitchat reply"


def test_graph_builds_with_dependencies(deps: AgentDependencies):
    graph = build_agent_graph(deps)
    result = graph.invoke({"user_id": "user4", "question": "my name is Alex"})
    assert result["route"] == "memory"


def _build_test_app(pipeline: AgenticPipeline) -> TestClient:
    app = FastAPI()

    @app.post("/api/ask")
    def ask(payload: Dict[str, str]):  # type: ignore[name-defined]
        user_id = payload.get("user_id")
        question = payload.get("question", "")
        result = pipeline.ask(user_id, question)
        return {"answer": result["answer"], "context": result["context"]}

    return TestClient(app)


def test_fastapi_endpoint_memory_flow(deps: AgentDependencies):
    pipeline = AgenticPipeline(deps)
    client = _build_test_app(pipeline)

    response = client.post("/api/ask", json={"user_id": "u10", "question": "My name is Jamie"})
    assert response.status_code == 200
    data = response.json()
    assert "remember" in data["answer"].lower()

    response = client.post("/api/ask", json={"user_id": "u10", "question": "what's my name"})
    assert response.status_code == 200
    data = response.json()
    assert "Jamie" in data["answer"]


def test_fastapi_endpoint_rag_flow(deps: AgentDependencies):
    pipeline = AgenticPipeline(deps)
    client = _build_test_app(pipeline)

    response = client.post(
        "/api/ask",
        json={"user_id": "u20", "question": "What are good sources of protein?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "RAG" in data["answer"]
    assert data["context"]


def test_fastapi_endpoint_chitchat_flow(deps: AgentDependencies):
    pipeline = AgenticPipeline(deps)
    client = _build_test_app(pipeline)

    response = client.post("/api/ask", json={"user_id": "u30", "question": "Hello!"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Chitchat reply"


def test_rag_with_personal_info_extraction(deps: AgentDependencies):
    """Test that personal information is extracted and stored even when RAG is triggered"""
    pipeline = AgenticPipeline(deps)
    store = deps.memory_store  # type: ignore[assignment]

    # This should trigger RAG (because of "fish" nutrition keyword) but also store preference
    response = pipeline.ask("user4", "I love to eat fish")
    assert response["route"] == "rag"
    assert "RAG" in response["answer"]  # Should get RAG response
    assert store.get("user4", "likes") == "fish"  # But should also store preference


def test_chitchat_with_personal_info_extraction(deps: AgentDependencies):
    """Test that personal information is extracted and stored even in chitchat"""
    pipeline = AgenticPipeline(deps)
    store = deps.memory_store  # type: ignore[assignment]

    # This should trigger chitchat but also store preference
    response = pipeline.ask("user5", "I really enjoy eating fish")
    assert response["route"] == "chitchat"
    assert store.get("user5", "likes") == "fish"  # Should store preference
