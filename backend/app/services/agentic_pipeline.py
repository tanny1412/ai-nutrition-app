from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, TypedDict, Tuple

from langchain.schema import HumanMessage
from langchain_openai import ChatOpenAI

try:  # pragma: no cover - exercised in environments with langgraph available
    from langgraph.graph import END, StateGraph
except ModuleNotFoundError:  # pragma: no cover - fallback for test environments without langgraph
    class _EndSentinel:
        pass

    END = _EndSentinel()

    class StateGraph:
        def __init__(self, state_type):  # noqa: D401 - simple fallback
            self._nodes = {}
            self._entry: Optional[str] = None
            self._edges: Dict[str, object] = {}
            self._conditional: Dict[str, tuple] = {}

        def add_node(self, name, func) -> None:
            self._nodes[name] = func

        def set_entry_point(self, name) -> None:
            self._entry = name

        def add_edge(self, start, end) -> None:
            self._edges[start] = end

        def add_conditional_edges(self, node, condition, mapping) -> None:
            self._conditional[node] = (condition, mapping)

        def compile(self):
            graph = self

            class _Compiled:
                def invoke(self_inner, state):
                    if graph._entry is None:
                        raise RuntimeError("Graph entry point not set")
                    current = graph._entry
                    result = dict(state)
                    while True:
                        node_fn = graph._nodes[current]
                        update = node_fn(result)
                        if update:
                            result.update(update)
                        if current in graph._conditional:
                            condition, mapping = graph._conditional[current]
                            route = condition(result)
                            next_node = mapping.get(route)
                            if next_node is None:
                                raise RuntimeError(f"No edge for route '{route}'")
                            current = next_node
                            if current is END:
                                break
                            continue
                        if current in graph._edges:
                            next_node = graph._edges[current]
                            if next_node is END:
                                break
                            current = next_node
                            continue
                        break
                    return result

            return _Compiled()

from ..config import get_settings
from .memory_service import MemoryService, get_memory_service
from .rag_service import RAGService, get_rag_service


class ChatLLM(Protocol):
    def __call__(self, question: str) -> str:  # pragma: no cover - protocol
        ...


class KeyValueStore(Protocol):
    def set(self, user_id: str, key: str, value: str) -> None:  # pragma: no cover - protocol
        ...

    def get(self, user_id: str, key: str) -> Optional[str]:  # pragma: no cover - protocol
        ...

    def as_dict(self, user_id: str) -> Dict[str, str]:  # pragma: no cover - protocol
        ...


class SQLKeyValueStore:
    def __init__(self, service: MemoryService) -> None:
        self._service = service

    def set(self, user_id: str, key: str, value: str) -> None:
        self._service.set_user_memory(user_id, key, value)

    def get(self, user_id: str, key: str) -> Optional[str]:
        return self._service.get_user_memory(user_id, key)

    def as_dict(self, user_id: str) -> Dict[str, str]:
        return self._service.get_all_user_memory(user_id)


class InMemoryKeyValueStore:
    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, str]] = {}

    def set(self, user_id: str, key: str, value: str) -> None:
        self._data.setdefault(user_id, {})[key] = value

    def get(self, user_id: str, key: str) -> Optional[str]:
        return self._data.get(user_id, {}).get(key)

    def as_dict(self, user_id: str) -> Dict[str, str]:
        return dict(self._data.get(user_id, {}))


@dataclass
class AgentDependencies:
    rag_service: RAGService
    memory_store: KeyValueStore
    chat_llm: ChatLLM


class AgentState(TypedDict, total=False):
    user_id: str
    question: str
    route: str
    answer: str
    context: List[Dict[str, object]]


NUTRITION_KEYWORDS = {
    "protein",
    "macro",
    "macros",
    "calorie",
    "calories",
    "diet",
    "nutrition",
    "nutrient",
    "meal",
    "carb",
    "carbs",
    "fat",
    "fiber",
    "vegetarian",
    "vegan",
    "hydration",
    "supplement",
    "vitamin",
}

# Removed rigid patterns - now using LLM-driven flexible memory system


def classify_intent(question: str, llm: Optional[ChatLLM] = None) -> str:
    # Always try LLM classification first for maximum flexibility
    if llm is not None and not isinstance(llm, (type(lambda x: x),)):
        try:
            prompt = (
                "You are an intent classifier for a nutrition assistant.\n"
                "Label the user message as one of: memory, rag, chitchat.\n"
                "- memory: user states ANY personal fact about themselves OR asks to recall personal information (name, preferences, goals, allergies, anything personal)\n"
                "- rag: nutrition/diet/food-related knowledge questions that need factual information\n"
                "- chitchat: greetings and small talk\n"
                f"Message: {question}\nReturn only the label."
            )
            label = llm(prompt).strip().lower()
            if label in {"memory", "rag", "chitchat"}:
                return label
        except Exception:
            pass
    
    # Simple fallback - if it has nutrition keywords, it's probably RAG
    text = question.lower().strip()
    if any(keyword in text for keyword in NUTRITION_KEYWORDS):
        return "rag"
    return "chitchat"


def _llm_extract_memory(question: str, llm: ChatLLM) -> Optional[Dict[str, str]]:
    """Extract any personal information from user message using LLM"""
    try:
        prompt = (
            "Extract personal information from this message. Return JSON with keys and values.\n"
            "Examples of keys: name, likes, dislikes, allergies, goals, preferences, dietary_restrictions, etc.\n"
            "If the user mentions multiple things, include all of them.\n"
            "If no personal information, return empty JSON: {}\n"
            f"Message: {question}\n"
            "Respond with valid JSON only."
        )
        response = llm(prompt).strip()
        
        # Clean up response to extract JSON
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        import json
        data = json.loads(response)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _llm_answer_from_memory(question: str, memory_dict: Dict[str, str], llm: ChatLLM) -> str:
    """Use LLM to answer questions using stored memory"""
    if not memory_dict:
        return "I don't have any personal information about you stored yet."
    
    try:
        facts = "; ".join(f"{k}: {v}" for k, v in memory_dict.items())
        prompt = (
            "Answer the user's question using ONLY the personal information provided.\n"
            f"Personal information: {facts}\n"
            f"Question: {question}\n"
            "If the information doesn't contain the answer, say 'I don't have that information about you yet.'\n"
            "Be conversational and helpful."
        )
        return llm(prompt)
    except Exception:
        return "I don't have that information about you yet."

def build_agent_graph(deps: AgentDependencies):
    graph = StateGraph(AgentState)

    def classifier_node(state: AgentState) -> AgentState:
        route = classify_intent(state["question"], deps.chat_llm)
        return {"route": route}

    def route_selector(state: AgentState) -> str:
        return state.get("route", "chitchat")

    def memory_node(state: AgentState) -> AgentState:
        user_id = state["user_id"]
        question = state["question"]
        
        # Get all stored memory for this user
        memory_dict = deps.memory_store.as_dict(user_id)
        
        # Check if this is a question about personal info (using LLM)
        is_question = False
        try:
            prompt = (
                "Is this message asking about personal information about the user?\n"
                "Examples: 'what's my name?', 'what do I like?', 'what are my preferences?', 'tell me about myself'\n"
                f"Message: {question}\n"
                "Answer: yes or no"
            )
            response = deps.chat_llm(prompt).strip().lower()
            is_question = "yes" in response
        except Exception:
            pass
        
        if is_question:
            # User is asking about their stored information
            answer = _llm_answer_from_memory(question, memory_dict, deps.chat_llm)
            return {"route": "memory", "answer": answer, "context": []}
        
        # User is providing new personal information
        try:
            extracted = _llm_extract_memory(question, deps.chat_llm)
            if extracted:
                # Store all extracted information
                for key, value in extracted.items():
                    if key and value:
                        # For list-like keys (likes, dislikes, etc.), append to existing
                        if key in ["likes", "dislikes", "allergies", "goals", "preferences"]:
                            current = memory_dict.get(key, "")
                            if current:
                                # Simple append with comma separation
                                new_value = f"{current}, {value}"
                            else:
                                new_value = value
                            deps.memory_store.set(user_id, key, new_value)
                        else:
                            deps.memory_store.set(user_id, key, value)
                
                # Generate a natural response
                try:
                    prompt = (
                        "The user just shared personal information with you. Acknowledge it naturally and briefly.\n"
                        f"Information shared: {extracted}\n"
                        "Be friendly and conversational. Don't repeat everything back."
                    )
                    answer = deps.chat_llm(prompt)
                    return {"route": "memory", "answer": answer, "context": []}
                except Exception:
                    return {"route": "memory", "answer": "Got it, I'll remember that about you!", "context": []}
        except Exception:
            pass
        
        return {"route": "memory", "answer": "I didn't catch any personal information, but I'm here to help!", "context": []}

    def rag_node(state: AgentState) -> AgentState:
        user_id = state["user_id"]
        question = state["question"]
        
        # Always check for personal information even in RAG responses
        memory_dict = deps.memory_store.as_dict(user_id)
        extracted = _llm_extract_memory(question, deps.chat_llm)
        
        # Store any personal information found
        if extracted:
            for key, value in extracted.items():
                if key and value:
                    # For list-like keys (likes, dislikes, etc.), append to existing
                    if key in ["likes", "dislikes", "allergies", "goals", "preferences"]:
                        current = memory_dict.get(key, "")
                        if current:
                            new_value = f"{current}, {value}"
                        else:
                            new_value = value
                        deps.memory_store.set(user_id, key, new_value)
                    else:
                        deps.memory_store.set(user_id, key, value)
        
        # Get RAG response
        result = deps.rag_service.answer(question)
        answer = result.get("answer", "I couldn't find relevant information right now.")
        context = result.get("context", [])
        
        # If we found personal information, enhance the response
        if extracted:
            try:
                # Create a more personalized response that acknowledges the preference
                personal_info = ", ".join(f"{k}: {v}" for k, v in extracted.items())
                enhanced_prompt = (
                    f"The user just mentioned: {personal_info}. "
                    f"Provide a nutrition response that acknowledges their preference: {answer}"
                )
                enhanced_answer = deps.chat_llm(enhanced_prompt)
                return {
                    "route": "rag",
                    "answer": enhanced_answer,
                    "context": context,
                }
            except Exception:
                # Fallback to original answer if enhancement fails
                pass
        
        return {
            "route": "rag",
            "answer": answer,
            "context": context,
        }

    def chitchat_node(state: AgentState) -> AgentState:
        user_id = state["user_id"]
        question = state["question"]
        
        # Always check for personal information even in chitchat
        memory_dict = deps.memory_store.as_dict(user_id)
        extracted = _llm_extract_memory(question, deps.chat_llm)
        
        # Store any personal information found
        if extracted:
            for key, value in extracted.items():
                if key and value:
                    # For list-like keys (likes, dislikes, etc.), append to existing
                    if key in ["likes", "dislikes", "allergies", "goals", "preferences"]:
                        current = memory_dict.get(key, "")
                        if current:
                            new_value = f"{current}, {value}"
                        else:
                            new_value = value
                        deps.memory_store.set(user_id, key, new_value)
                    else:
                        deps.memory_store.set(user_id, key, value)
        
        # Get chitchat response
        answer = deps.chat_llm(question)
        
        # If we found personal information, enhance the response
        if extracted:
            try:
                # Create a more personalized response that acknowledges the preference
                personal_info = ", ".join(f"{k}: {v}" for k, v in extracted.items())
                enhanced_prompt = (
                    f"The user just mentioned: {personal_info}. "
                    f"Respond naturally while acknowledging their preference: {answer}"
                )
                enhanced_answer = deps.chat_llm(enhanced_prompt)
                return {
                    "route": "chitchat",
                    "answer": enhanced_answer,
                    "context": [],
                }
            except Exception:
                # Fallback to original answer if enhancement fails
                pass
        
        return {
            "route": "chitchat",
            "answer": answer,
            "context": [],
        }

    graph.add_node("classifier", classifier_node)
    graph.add_node("memory", memory_node)
    graph.add_node("rag", rag_node)
    graph.add_node("chitchat", chitchat_node)

    graph.set_entry_point("classifier")
    graph.add_conditional_edges(
        "classifier",
        route_selector,
        {
            "memory": "memory",
            "rag": "rag",
            "chitchat": "chitchat",
        },
    )
    graph.add_edge("memory", END)
    graph.add_edge("rag", END)
    graph.add_edge("chitchat", END)

    return graph.compile()


class AgenticPipeline:
    def __init__(self, deps: AgentDependencies) -> None:
        self._graph = build_agent_graph(deps)
        self._deps = deps

    def ask(self, user_id: Optional[str], question: str) -> Dict[str, object]:
        uid = user_id or "anonymous"
        state: AgentState = {
            "user_id": uid,
            "question": question,
        }
        result = self._graph.invoke(state)
        return {
            "answer": result.get("answer", ""),
            "context": result.get("context", []),
            "route": result.get("route", "chitchat"),
        }


class OpenAIChatWrapper:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo", temperature: float = 0.6) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for chitchat responses.")
        self._llm = ChatOpenAI(api_key=api_key, model=model, temperature=temperature)

    def __call__(self, question: str) -> str:
        response = self._llm.invoke([HumanMessage(content=question)])
        return response.content


def get_agent_pipeline() -> AgenticPipeline:
    if hasattr(get_agent_pipeline, "_instance"):
        return get_agent_pipeline._instance  # type: ignore[attr-defined]

    settings = get_settings()
    rag_service = get_rag_service()
    memory_service = get_memory_service()
    if memory_service:
        memory_store: KeyValueStore = SQLKeyValueStore(memory_service)
    else:
        memory_store = InMemoryKeyValueStore()

    chat_llm: ChatLLM
    try:
        chat_llm = OpenAIChatWrapper(api_key=settings.openai_api_key, model=settings.chat_model)
    except RuntimeError:
        # Fallback simple echo for environments without API key
        chat_llm = lambda question: "Let's focus on nutrition questions!"  # type: ignore[assignment]

    deps = AgentDependencies(
        rag_service=rag_service,
        memory_store=memory_store,
        chat_llm=chat_llm,
    )
    get_agent_pipeline._instance = AgenticPipeline(deps)  # type: ignore[attr-defined]
    return get_agent_pipeline._instance  # type: ignore[attr-defined]


__all__ = [
    "AgenticPipeline",
    "AgentDependencies",
    "InMemoryKeyValueStore",
    "SQLKeyValueStore",
    "classify_intent",
    "build_agent_graph",
    "get_agent_pipeline",
]
