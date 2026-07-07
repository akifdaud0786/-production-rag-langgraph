"""Shared state schema passed between every LangGraph node."""
from __future__ import annotations

from typing import Annotated, List, Optional, TypedDict

from langgraph.graph.message import add_messages


class RetrievedChunkDict(TypedDict):
    chunk_id: str
    text: str
    score: float
    metadata: dict


class RAGState(TypedDict, total=False):
    # Conversation
    messages: Annotated[list, add_messages]

    # Query understanding
    raw_query: str
    standalone_query: str  # history-aware, de-referenced query

    # Retrieval
    retrieved_chunks: List[RetrievedChunkDict]
    reranked_chunks: List[RetrievedChunkDict]

    # Generation
    draft_answer: str
    final_answer: str

    # Critique / evaluate loop
    is_satisfactory: bool
    critique_feedback: str
    refine_count: int
    max_refine_iterations: int

    # Observability
    trace: List[dict]
