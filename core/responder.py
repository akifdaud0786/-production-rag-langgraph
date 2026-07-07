"""Responder: "3. Generate" node — produces a grounded answer from re-ranked chunks."""
from __future__ import annotations

from typing import List

from core.state import RetrievedChunkDict
from gateway.llm_gateway import get_gateway

_SYSTEM_PROMPT = """You are a precise technical assistant in a Retrieval-Augmented
Generation system. Answer the user's question using ONLY the provided context
chunks ("true data"). If the context does not contain enough information to
answer confidently, say so explicitly rather than guessing. Cite which chunk(s)
support each claim using their chunk_id in square brackets, e.g. [chunk_3].
Do not fabricate information outside the given context."""


def _format_context(chunks: List[RetrievedChunkDict]) -> str:
    if not chunks:
        return "(no relevant context retrieved)"
    return "\n\n".join(f"[{c['chunk_id']}] (relevance={c['score']:.2f})\n{c['text']}" for c in chunks)


def generate_answer(query: str, chunks: List[RetrievedChunkDict], feedback: str = "") -> str:
    gateway = get_gateway()
    context = _format_context(chunks)

    user_content = f"Context:\n{context}\n\nQuestion: {query}"
    if feedback:
        user_content += f"\n\nNote: a previous answer attempt was rejected for this reason — address it: {feedback}"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    response = gateway.call(messages, temperature=0.2, max_tokens=800)
    return response.text.strip()
