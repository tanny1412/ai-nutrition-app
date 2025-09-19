from typing import List, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., description="User question about nutrition")
    top_k: Optional[int] = Field(
        default=None,
        description="Override the number of documents to retrieve from the knowledge base.",
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
