"""
Retriever: "2. Retrieve" node, plus the FlashRank-based semantic re-ranker
that separates true (relevant, technical) data from noisy data.

Flow: vector similarity search (recall-oriented, top ~20) -> cross-encoder
re-rank against the standalone query (precision-oriented) -> drop anything
below RERANK_MIN_SCORE -> keep top RERANK_TOP_K.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import List

from core.state import RetrievedChunkDict
from core.vector_store import get_vector_store

VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "20"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RERANK_MIN_SCORE = float(os.getenv("RERANK_MIN_SCORE", "0.35"))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")


@lru_cache(maxsize=1)
def _get_reranker():
    from flashrank import Ranker
    from pathlib import Path

    cache_dir = str(Path(__file__).resolve().parent.parent / "flashrank_cache")
    return Ranker(model_name=RERANKER_MODEL, cache_dir=cache_dir)


def retrieve(query: str, top_k: int = VECTOR_TOP_K) -> List[RetrievedChunkDict]:
    store = get_vector_store()
    hits = store.search(query, top_k=top_k)
    return [
        RetrievedChunkDict(chunk_id=h.chunk_id, text=h.text, score=h.score, metadata=h.metadata)
        for h in hits
    ]


def rerank(query: str, chunks: List[RetrievedChunkDict], top_k: int = RERANK_TOP_K) -> List[RetrievedChunkDict]:
    """Cross-encoder re-rank against the query; filters out noisy/off-topic chunks."""
    if not chunks:
        return []

    from flashrank import RerankRequest

    ranker = _get_reranker()
    passages = [{"id": c["chunk_id"], "text": c["text"], "meta": c["metadata"]} for c in chunks]
    request = RerankRequest(query=query, passages=passages)
    results = ranker.rerank(request)

    reranked: List[RetrievedChunkDict] = []
    for r in results:
        if r["score"] < RERANK_MIN_SCORE:
            continue  # this is "noisy data" — drop it before it reaches generation
        reranked.append(
            RetrievedChunkDict(
                chunk_id=r["id"],
                text=r["text"],
                score=float(r["score"]),
                metadata=r.get("meta", {}),
            )
        )
        if len(reranked) >= top_k:
            break
    return reranked
