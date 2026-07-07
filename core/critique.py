"""
Critique & Evaluate: "4. Critique & Evaluate" node.

A lightweight LLM-as-judge checks the draft answer against the retrieved
chunks for groundedness and completeness. This is what drives the cyclic
"Satisfactory Answer? Yes/No" decision in the workflow diagram.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

from core.state import RetrievedChunkDict
from gateway.llm_gateway import get_gateway

CRITIQUE_MODEL = os.getenv("CRITIQUE_MODEL", "llama-3.1-8b-instant")

_SYSTEM_PROMPT = """You are a strict evaluator (LLM judge) for a RAG system.
Given the question, the retrieved context chunks, and a draft answer, decide
if the answer is satisfactory. An answer is satisfactory ONLY if:
  1. It is fully grounded in the provided context (no fabricated claims).
  2. It directly and substantively answers the question.
  3. It does not rely on irrelevant/noisy chunks that don't pertain to the question.

Respond with STRICT JSON only, no markdown, matching this schema exactly:
{"is_satisfactory": true|false, "feedback": "<one or two sentence reason, and if not satisfactory, what to search for instead>"}"""


@dataclass
class CritiqueResult:
    is_satisfactory: bool
    feedback: str


def critique_answer(query: str, chunks: List[RetrievedChunkDict], draft_answer: str) -> CritiqueResult:
    gateway = get_gateway()
    context = "\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in chunks) or "(no context retrieved)"

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Question: {query}\n\nContext:\n{context}\n\nDraft answer:\n{draft_answer}",
        },
    ]
    response = gateway.call(messages, temperature=0.0, max_tokens=250, model=CRITIQUE_MODEL)

    try:
        cleaned = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        return CritiqueResult(
            is_satisfactory=bool(parsed.get("is_satisfactory", False)),
            feedback=str(parsed.get("feedback", "")),
        )
    except (json.JSONDecodeError, AttributeError):
        # Fail safe: if the judge's output can't be parsed, don't loop forever —
        # treat as satisfactory but flag it in the feedback for observability.
        return CritiqueResult(is_satisfactory=True, feedback="critique_parse_failed_defaulting_to_pass")
