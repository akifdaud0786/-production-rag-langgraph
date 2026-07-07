"""
Embedding layer. Uses a local sentence-transformers model by default so the
ingestion pipeline doesn't require an external embedding API call — keeping
cost and latency low for enterprise-scale corpora.
"""
from __future__ import annotations

import os
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_embedder() -> HuggingFaceEmbeddings:
    """Singleton embedder so the model is only loaded into memory once."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    return get_embedder().embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return get_embedder().embed_query(text)
