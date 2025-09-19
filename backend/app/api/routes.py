from fastapi import APIRouter, Depends, HTTPException


from ..schemas import (
    AnswerResponse,
    IngestRequest,
    IngestUrlRequest,
    QuestionRequest,
    RetrievedChunk,
)
from ..services.rag_service import RAGService, get_rag_service

router = APIRouter()


@router.get("/health")
def healthcheck() -> dict:
    return {"status": "ok"}


@router.post("/ask", response_model=AnswerResponse)
def ask_question(
    payload: QuestionRequest,
    rag: RAGService = Depends(get_rag_service),
) -> AnswerResponse:
    try:
        result = rag.answer(payload.question, top_k=payload.top_k)
    except RuntimeError as exc:  # Typically missing API keys or no documents
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return AnswerResponse(
        answer=result["answer"],
        context=[RetrievedChunk(**chunk) for chunk in result["context"]],
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
