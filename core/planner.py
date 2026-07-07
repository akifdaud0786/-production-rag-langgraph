"""
Planner: "1. Query Understanding" and "5. Refine Query" nodes.

History-aware reformulation — takes raw user query + prior conversation
turns and produces a standalone query with pronouns/references resolved,
so retrieval isn't polluted by ambiguous follow-ups like "what about that?".
"""
from __future__ import annotations

from gateway.llm_gateway import get_gateway

_QUERY_UNDERSTANDING_SYSTEM = """You are a query rewriting module in a RAG system.
Given the conversation history and the latest user message, rewrite the latest
message into a standalone, fully self-contained question. Resolve pronouns and
references using the history. Do not answer the question. Output ONLY the
rewritten question, nothing else."""

_REFINE_QUERY_SYSTEM = """You are a query refinement module in a RAG system.
The previous retrieval + answer attempt was judged NOT satisfactory for the
reason given below. Rewrite the standalone query to retrieve better, more
specific "true data" and avoid the noisy/irrelevant results that caused the
problem. Output ONLY the rewritten query, nothing else."""


def understand_query(raw_query: str, conversation_history: str = "") -> str:
    gateway = get_gateway()
    user_content = f"Conversation history:\n{conversation_history or '(none)'}\n\nLatest message: {raw_query}"
    messages = [
        {"role": "system", "content": _QUERY_UNDERSTANDING_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    response = gateway.call(messages, temperature=0.0, max_tokens=200)
    return response.text.strip().strip('"')


def refine_query(standalone_query: str, critique_feedback: str) -> str:
    gateway = get_gateway()
    messages = [
        {"role": "system", "content": _REFINE_QUERY_SYSTEM},
        {
            "role": "user",
            "content": f"Original query: {standalone_query}\n\nWhy it failed: {critique_feedback}\n\nRewrite the query:",
        },
    ]
    response = gateway.call(messages, temperature=0.3, max_tokens=200)
    return response.text.strip().strip('"')
