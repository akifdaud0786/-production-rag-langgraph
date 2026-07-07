from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    thread_id: str = Field(default="default", description="Conversation/session id for memory continuity")


class ChunkResponse(BaseModel):
    chunk_id: str
    score: float
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    standalone_query: str
    refine_count: int
    is_satisfactory: bool
    sources: List[ChunkResponse]
    trace: Optional[List[dict]] = None


class HealthResponse(BaseModel):
    status: str
    vector_backend: str
