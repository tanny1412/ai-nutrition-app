from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException


from ..schemas import (
    AnswerResponse,
    ConversationHistoryResponse,
    ConversationMetadata,
    ConversationMessage,
    CreateSessionRequest,
    CreateSessionResponse,
    IngestRequest,
    IngestUrlRequest,
    QuestionRequest,
    RetrievedChunk,
)
from ..services.simple_agentic_pipeline import SimpleAgenticPipeline, get_simple_agent_pipeline
from ..services.memory_service import get_memory_service
from ..services.rag_service import RAGService, get_rag_service

router = APIRouter()


@router.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@router.post("/ask", response_model=AnswerResponse)
def ask_question(
    payload: QuestionRequest,
    agent: SimpleAgenticPipeline = Depends(get_simple_agent_pipeline),
) -> AnswerResponse:
    try:
        result = agent.ask(
            payload.user_id,
            payload.question,
            session_id=payload.session_id,
            top_k=payload.top_k,
        )
    except RuntimeError as exc:  # Typically missing API keys or no documents
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnswerResponse(
        answer=result["answer"],
        context=[RetrievedChunk(**chunk) for chunk in result.get("context", [])],
    )


@router.post("/ingest")
def ingest_document(
    payload: IngestRequest,
    rag: RAGService = Depends(get_rag_service),
) -> dict:
    try:
        added_chunks = rag.ingest(payload.content, title=payload.title)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"chunks_indexed": added_chunks}



@router.post("/ingest_url")
def ingest_url(
    payload: IngestUrlRequest,
    rag: RAGService = Depends(get_rag_service),
) -> dict:
    try:
        added_chunks = rag.ingest_url(payload.url, title=payload.title)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"chunks_indexed": added_chunks}


def _require_memory_service():
    memory = get_memory_service()
    if memory is None:
        raise HTTPException(status_code=503, detail="Conversational memory is disabled.")
    return memory


@router.post("/sessions/create", response_model=CreateSessionResponse)
def create_session(payload: CreateSessionRequest) -> CreateSessionResponse:
    memory = _require_memory_service()
    session_id = memory.create_conversation(user_id=payload.user_id, title=payload.title)
    return CreateSessionResponse(session_id=session_id)


@router.get("/sessions/list", response_model=List[ConversationMetadata])
def list_sessions(user_id: Optional[str] = None) -> List[ConversationMetadata]:
    memory = _require_memory_service()
    conversations = memory.list_conversations(user_id=user_id)
    payload: List[ConversationMetadata] = []
    for convo in conversations:
        payload.append(
            ConversationMetadata(
                id=convo.id,
                title=convo.title,
                user_id=convo.user_id,
                created_at=convo.created_at.isoformat(),
                updated_at=convo.updated_at.isoformat(),
            ),
        )
    return payload


@router.get("/sessions/{session_id}/history", response_model=ConversationHistoryResponse)
def get_history(session_id: str) -> ConversationHistoryResponse:
    memory = _require_memory_service()
    if not memory.conversation_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    summary = memory.get_summary(session_id)
    messages = memory.iter_messages(session_id)
    response_messages = [
        ConversationMessage(role=msg.role, content=msg.content, created_at=msg.created_at.isoformat())
        for msg in messages
    ]
    return ConversationHistoryResponse(summary=summary, messages=response_messages)


@router.post("/sessions/{session_id}/reset")
def reset_session(session_id: str) -> dict:
    memory = _require_memory_service()
    if not memory.conversation_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    memory.reset_conversation(session_id)
    return {"ok": True}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    memory = _require_memory_service()
    if not memory.conversation_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    memory.delete_conversation(session_id)
    return {"ok": True}


@router.get("/conversation/{user_id}/history")
def get_conversation_history(
    user_id: str,
    agent: SimpleAgenticPipeline = Depends(get_simple_agent_pipeline),
) -> dict:
    """Get conversation history for a user using the simple agent pipeline"""
    try:
        history = agent.get_conversation_history(user_id)
        return {"history": history}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/conversation/{user_id}")
def clear_conversation_history(
    user_id: str,
    agent: SimpleAgenticPipeline = Depends(get_simple_agent_pipeline),
) -> dict:
    """Clear conversation history for a user"""
    try:
        agent.clear_memory(user_id)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
