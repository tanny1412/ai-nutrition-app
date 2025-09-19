from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., description="User question about nutrition")
    top_k: Optional[int] = Field(
        default=None,
        description="Override the number of documents to retrieve from the knowledge base.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Conversation/session identifier for follow-up context.",
    )
    user_id: Optional[str] = Field(
        default=None,
        description="Optional user identifier used to scope sessions.",
    )


class RetrievedChunk(BaseModel):
    content: str
    metadata: dict


class AnswerResponse(BaseModel):
    answer: str
    context: List[RetrievedChunk]


class IngestRequest(BaseModel):
    content: str = Field(..., description="Plain-text document content to index")
    title: Optional[str] = Field(
        default=None,
        description="Optional label used to identify the uploaded document.",
    )


class IngestUrlRequest(BaseModel):
    url: str = Field(..., description="Absolute URL to fetch and ingest")
    title: Optional[str] = Field(
        default=None,
        description="Optional title to associate with the fetched content.",
    )


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = Field(default=None)
    title: Optional[str] = Field(default=None)


class CreateSessionResponse(BaseModel):
    session_id: str


class ConversationMessage(BaseModel):
    role: str
    content: str
    created_at: str


class ConversationHistoryResponse(BaseModel):
    summary: Optional[str]
    messages: List[ConversationMessage]


class ConversationMetadata(BaseModel):
    id: str
    title: Optional[str]
    user_id: Optional[str]
    created_at: str
    updated_at: str
