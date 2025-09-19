from __future__ import annotations

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

from ..config import get_settings
from .rag_service import RAGService, get_rag_service


@dataclass
class AgentResponse:
    answer: str
    context: List[Dict[str, Any]]
    route: str


class SimpleAgenticPipeline:
    """
    A simplified agentic pipeline using LangChain's built-in memory.
    This replaces the complex custom memory system with LangChain's proven memory modules.
    """
    
    def __init__(self, rag_service: RAGService, memory_type: str = "buffer"):
        self.rag_service = rag_service
        self.settings = get_settings()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=self.settings.openai_api_key,
            model=self.settings.chat_model,
            temperature=self.settings.temperature
        )
        
        # Initialize memory - one per user
        self._memories: Dict[str, Any] = {}
        self.memory_type = memory_type
        
        # Create the main prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful nutrition assistant. You have access to:
1. The user's conversation history (which you can see in the chat history)
2. A knowledge base of nutrition information (which you can search)

Your capabilities:
- Answer nutrition and diet-related questions using your knowledge base
- Remember and reference information the user has shared about themselves
- Provide personalized advice based on what you know about the user
- Have natural conversations about food, health, and nutrition

Guidelines:
- Always be helpful, accurate, and encouraging
- If you don't know something, say so and offer to help find the information
- Reference the user's preferences and past conversations when relevant
- Keep responses concise but informative"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm
    
    def _get_memory(self, user_id: str):
        """Get or create memory for a user"""
        if user_id not in self._memories:
            if self.memory_type == "summary":
                self._memories[user_id] = ConversationSummaryMemory(
                    llm=self.llm,
                    return_messages=True,
                    memory_key="chat_history"
                )
            else:  # buffer
                self._memories[user_id] = ConversationBufferMemory(
                    return_messages=True,
                    memory_key="chat_history"
                )
        return self._memories[user_id]
    
    def _classify_intent(self, question: str) -> str:
        """Classify the user's intent"""
        question_lower = question.lower()
        
        # Check for personal information patterns
        personal_patterns = [
            "my name is", "i am", "i like", "i love", "i prefer", "i hate", "i dislike",
            "i'm allergic", "my goal", "i want to", "i need to", "i'm trying to"
        ]
        
        if any(pattern in question_lower for pattern in personal_patterns):
            return "personal"
        
        # Check for questions about personal info
        personal_questions = [
            "what's my", "what is my", "what do i", "what are my", "tell me about myself",
            "what did i say", "what did i tell you", "do you remember"
        ]
        
        if any(pattern in question_lower for pattern in personal_questions):
            return "personal"
        
        # Check for nutrition-related keywords
        nutrition_keywords = [
            "protein", "calories", "nutrition", "diet", "vitamin", "mineral",
            "carbohydrate", "fat", "fiber", "meal", "recipe", "healthy",
            "weight", "exercise", "supplement", "allergy", "intolerance"
        ]
        
        if any(keyword in question_lower for keyword in nutrition_keywords):
            return "nutrition"
        
        return "general"
    
    def _enhance_with_rag(self, question: str, base_response: str) -> tuple[str, List[Dict[str, Any]]]:
        """Enhance the response with RAG information if needed"""
        try:
            rag_result = self.rag_service.answer(question)
            rag_answer = rag_result.get("answer", "")
            rag_context = rag_result.get("context", [])
            
            if rag_answer and rag_answer != "I couldn't find relevant information right now.":
                # Combine the base response with RAG information
                enhanced_prompt = f"""
The user asked: {question}

Base response: {base_response}

Additional nutrition information: {rag_answer}

Please provide a comprehensive response that combines the base response with the additional nutrition information. Be natural and conversational.
"""
                enhanced_response = self.llm.invoke([HumanMessage(content=enhanced_prompt)]).content
                return enhanced_response, rag_context
            
        except Exception:
            pass
        
        return base_response, []
    
    def ask(self, user_id: Optional[str], question: str) -> Dict[str, Any]:
        """Main method to process user questions"""
        uid = user_id or "anonymous"
        
        # Get user's memory
        memory = self._get_memory(uid)
        
        # Classify intent
        intent = self._classify_intent(question)
        
        # Get conversation history
        chat_history = memory.chat_memory.messages
        
        # Generate response using the chain
        try:
            response = self.chain.invoke({
                "input": question,
                "chat_history": chat_history
            })
            base_answer = response.content
        except Exception as e:
            base_answer = f"I'm sorry, I encountered an error: {str(e)}"
        
        # Enhance with RAG if it's a nutrition question
        context = []
        if intent == "nutrition":
            base_answer, context = self._enhance_with_rag(question, base_answer)
        
        # Save the conversation
        memory.save_context(
            {"input": question},
            {"output": base_answer}
        )
        
        # Determine route
        if intent == "personal":
            route = "memory"
        elif intent == "nutrition":
            route = "rag"
        else:
            route = "chitchat"
        
        return {
            "answer": base_answer,
            "context": context,
            "route": route
        }
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a user"""
        if user_id not in self._memories:
            return []
        
        memory = self._memories[user_id]
        messages = memory.chat_memory.messages
        
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "human", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        
        return history
    
    def clear_memory(self, user_id: str):
        """Clear memory for a user"""
        if user_id in self._memories:
            del self._memories[user_id]


def get_simple_agent_pipeline() -> SimpleAgenticPipeline:
    """Get or create the simple agent pipeline instance"""
    if hasattr(get_simple_agent_pipeline, "_instance"):
        return get_simple_agent_pipeline._instance
    
    rag_service = get_rag_service()
    get_simple_agent_pipeline._instance = SimpleAgenticPipeline(rag_service)
    return get_simple_agent_pipeline._instance
