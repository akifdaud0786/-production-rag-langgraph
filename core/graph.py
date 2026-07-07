"""
The Cyclic RAG Workflow (LangGraph), matching the reference diagram exactly:

    1. Query Understanding -> 2. Retrieve -> 3. Generate -> 4. Critique & Evaluate
                                   ▲                              │
                                   └──── 5. Refine Query ◄── "No" ┘
                                                                   │ "Yes"
                                                              Final Answer

`MemorySaver` gives each conversation (keyed by thread_id) durable short-term
memory across turns, matching the "MemorySaver / Conversation history" box.
"""
from __future__ import annotations

import os
import logging

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from core.critique import critique_answer
from core.planner import refine_query, understand_query
from core.responder import generate_answer
from core.retriever import rerank, retrieve
from core.state import RAGState

logger = logging.getLogger("core.graph")
MAX_REFINE_ITERATIONS = int(os.getenv("MAX_REFINE_ITERATIONS", "2"))


def _history_as_text(state: RAGState) -> str:
    messages = state.get("messages", [])
    lines = []
    for m in messages[-6:]:  # last 3 turns
        role = getattr(m, "type", getattr(m, "role", "user"))
        content = getattr(m, "content", str(m))
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Node 1: Query Understanding
# ---------------------------------------------------------------------------
def node_query_understanding(state: RAGState) -> dict:
    history = _history_as_text(state)
    standalone = understand_query(state["raw_query"], history)
    logger.info("Node 1: Raw Query = %r | Standalone Query = %r", state["raw_query"], standalone)
    return {
        "standalone_query": standalone,
        "refine_count": 0,
        "max_refine_iterations": state.get("max_refine_iterations", MAX_REFINE_ITERATIONS),
        "trace": state.get("trace", []) + [{"node": "query_understanding", "output": standalone}],
    }


# ---------------------------------------------------------------------------
# Node 2: Retrieve (+ semantic re-rank -> true data vs noisy data)
# ---------------------------------------------------------------------------
def node_retrieve(state: RAGState) -> dict:
    query = state["standalone_query"]
    raw_hits = retrieve(query)
    reranked = rerank(query, raw_hits)
    logger.info("Node 2: Retrieve Query = %r | Raw Hits = %d | Kept After Rerank = %d", query, len(raw_hits), len(reranked))
    return {
        "retrieved_chunks": raw_hits,
        "reranked_chunks": reranked,
        "trace": state.get("trace", []) + [
            {"node": "retrieve", "raw_hits": len(raw_hits), "kept_after_rerank": len(reranked)}
        ],
    }


# ---------------------------------------------------------------------------
# Node 3: Generate
# ---------------------------------------------------------------------------
def node_generate(state: RAGState) -> dict:
    answer = generate_answer(
        state["standalone_query"],
        state.get("reranked_chunks", []),
        feedback=state.get("critique_feedback", ""),
    )
    logger.info("Node 3: Generated Draft Answer = %r", answer[:150] + "...")
    return {
        "draft_answer": answer,
        "trace": state.get("trace", []) + [{"node": "generate", "answer_preview": answer[:200]}],
    }


# ---------------------------------------------------------------------------
# Node 4: Critique & Evaluate
# ---------------------------------------------------------------------------
def node_critique(state: RAGState) -> dict:
    result = critique_answer(
        state["standalone_query"], state.get("reranked_chunks", []), state["draft_answer"]
    )
    logger.info("Node 4: Critique Result: is_satisfactory = %s | Feedback = %r", result.is_satisfactory, result.feedback)
    return {
        "is_satisfactory": result.is_satisfactory,
        "critique_feedback": result.feedback,
        "trace": state.get("trace", []) + [
            {"node": "critique", "is_satisfactory": result.is_satisfactory, "feedback": result.feedback}
        ],
    }


# ---------------------------------------------------------------------------
# Node 5: Refine Query
# ---------------------------------------------------------------------------
def node_refine_query(state: RAGState) -> dict:
    refined = refine_query(state["standalone_query"], state.get("critique_feedback", ""))
    logger.info("Node 5: Refine Query: original = %r | feedback = %r | refined = %r", state["standalone_query"], state.get("critique_feedback", ""), refined)
    return {
        "standalone_query": refined,
        "refine_count": state.get("refine_count", 0) + 1,
        "trace": state.get("trace", []) + [{"node": "refine_query", "refined_query": refined}],
    }


# ---------------------------------------------------------------------------
# Terminal node: Final Answer
# ---------------------------------------------------------------------------
def node_final_answer(state: RAGState) -> dict:
    logger.info("Node Final Answer: Output = %r", state["draft_answer"][:150] + "...")
    return {
        "final_answer": state["draft_answer"],
        "messages": [{"role": "assistant", "content": state["draft_answer"]}],
    }


def _route_after_critique(state: RAGState) -> str:
    """The 'Satisfactory Answer? Yes/No' decision diamond."""
    if state.get("is_satisfactory"):
        return "generate_final_answer"
    if state.get("refine_count", 0) >= state.get("max_refine_iterations", MAX_REFINE_ITERATIONS):
        # Stop looping even if not fully satisfactory — avoid infinite cycles / runaway cost.
        return "generate_final_answer"
    return "refine_query"


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("query_understanding", node_query_understanding)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("generate", node_generate)
    graph.add_node("critique", node_critique)
    graph.add_node("refine_query", node_refine_query)
    graph.add_node("generate_final_answer", node_final_answer)

    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critique")
    graph.add_conditional_edges(
        "critique",
        _route_after_critique,
        {"generate_final_answer": "generate_final_answer", "refine_query": "refine_query"},
    )
    graph.add_edge("refine_query", "retrieve")  # cyclic loop back to retrieval
    graph.add_edge("generate_final_answer", END)

    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_query(raw_query: str, thread_id: str = "default") -> RAGState:
    """Convenience wrapper used by the API layer."""
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial_state: RAGState = {
        "raw_query": raw_query,
        "messages": [{"role": "user", "content": raw_query}],
    }
    result = graph.invoke(initial_state, config=config)
    return result
