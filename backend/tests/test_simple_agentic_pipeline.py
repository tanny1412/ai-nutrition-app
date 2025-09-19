import pytest
from unittest.mock import Mock, patch

from backend.app.services.simple_agentic_pipeline import SimpleAgenticPipeline


class DummyRAGService:
    """Minimal RAG stub used by simple pipeline tests."""

    def __init__(self) -> None:
        self.calls = []

    def answer(self, question: str, top_k: int | None = None, session_id=None, user_id=None) -> dict:  # noqa: ARG002
        self.calls.append(
            {
                "question": question,
                "top_k": top_k,
                "session_id": session_id,
                "user_id": user_id,
            }
        )
        return {
            "answer": f"RAG: contextual answer for '{question}'",
            "context": [
                {
                    "content": "Protein sources include beans and lentils.",
                    "metadata": {"source": "test_doc.md"},
                }
            ],
        }


class TestSimpleAgenticPipeline:
    
    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for testing"""
        mock = Mock()
        mock_response = Mock()
        mock_response.content = "Test response"
        mock.invoke.return_value = mock_response
        
        # Also mock the chain invoke method
        mock_chain = Mock()
        mock_chain.invoke.return_value = mock_response
        return mock
    
    @pytest.fixture
    def pipeline(self, mock_llm):
        """Create a test pipeline with mocked dependencies"""
        rag_service = DummyRAGService()
        
        with patch('backend.app.services.simple_agentic_pipeline.ChatOpenAI') as mock_chat:
            mock_chat.return_value = mock_llm
            pipeline = SimpleAgenticPipeline(rag_service)
            pipeline.llm = mock_llm
            
            # Mock the chain
            mock_chain = Mock()
            mock_response = Mock()
            mock_response.content = "Test response"
            mock_chain.invoke.return_value = mock_response
            pipeline.chain = mock_chain
            
            return pipeline
    
    def test_ask_stores_conversation(self, pipeline):
        """Test that conversations are stored in memory"""
        response = pipeline.ask("user1", "Hello, my name is John")
        
        assert response["answer"] == "Test response"
        assert response["route"] == "memory"
        
        # Check that conversation was stored
        history = pipeline.get_conversation_history("user1")
        assert len(history) == 2  # user message + assistant response
        assert history[0]["role"] == "human"
        assert history[0]["content"] == "Hello, my name is John"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Test response"
    
    def test_intent_classification(self, pipeline):
        """Test intent classification"""
        # Personal information
        assert pipeline._classify_intent("My name is John") == "personal"
        assert pipeline._classify_intent("I like fish") == "personal"
        assert pipeline._classify_intent("What's my name?") == "personal"
        
        # Nutrition questions
        assert pipeline._classify_intent("What are good sources of protein?") == "nutrition"
        assert pipeline._classify_intent("How many calories in an apple?") == "nutrition"
        
        # General conversation
        assert pipeline._classify_intent("Hello there!") == "general"
        assert pipeline._classify_intent("How's the weather?") == "general"
    
    def test_rag_enhancement(self, pipeline):
        """Test that nutrition questions get enhanced with RAG"""
        # Mock the RAG service to return some information
        pipeline.rag_service.answer = Mock(return_value={
            "answer": "Fish is an excellent source of protein and omega-3 fatty acids.",
            "context": [{"content": "Fish nutrition facts", "metadata": {"source": "test"}}]
        })
        
        # Mock the enhanced response
        enhanced_response = Mock()
        enhanced_response.content = "Enhanced response with nutrition info"
        pipeline.llm.invoke.return_value = enhanced_response
        
        response = pipeline.ask("user1", "What are good sources of protein?")
        
        assert response["route"] == "rag"
        assert response["context"] == [{"content": "Fish nutrition facts", "metadata": {"source": "test"}}]
        # The RAG service should have been called
        pipeline.rag_service.answer.assert_called_once()
    
    def test_memory_persistence(self, pipeline):
        """Test that memory persists across multiple questions"""
        # First conversation
        pipeline.ask("user1", "My name is Alice")
        
        # Second conversation - should remember the name
        response = pipeline.ask("user1", "What did I tell you about myself?")
        
        # Check that both conversations are in memory
        history = pipeline.get_conversation_history("user1")
        assert len(history) == 4  # 2 conversations * 2 messages each
        
        # The assistant should have access to the full conversation history
        assert any("Alice" in msg["content"] for msg in history if msg["role"] == "human")
    
    def test_clear_memory(self, pipeline):
        """Test that memory can be cleared"""
        # Add some conversation
        pipeline.ask("user1", "Hello")
        
        # Clear memory
        pipeline.clear_memory("user1")
        
        # Check that memory is empty
        history = pipeline.get_conversation_history("user1")
        assert len(history) == 0
    
    def test_different_users_separate_memory(self, pipeline):
        """Test that different users have separate memory"""
        # User 1 conversation
        pipeline.ask("user1", "My name is Alice")
        
        # User 2 conversation
        pipeline.ask("user2", "My name is Bob")
        
        # Check that each user has their own memory
        history1 = pipeline.get_conversation_history("user1")
        history2 = pipeline.get_conversation_history("user2")
        
        assert len(history1) == 2
        assert len(history2) == 2
        assert "Alice" in history1[0]["content"]
        assert "Bob" in history2[0]["content"]
        assert "Alice" not in history2[0]["content"]
        assert "Bob" not in history1[0]["content"]
