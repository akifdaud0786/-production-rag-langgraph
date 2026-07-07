"""
Chunking & embedding-ready splitting.

Uses a recursive character splitter tuned for technical documentation, with
overlap to preserve cross-chunk context — important for distinguishing
"true data" (dense technical passages) from "noisy data" (short boilerplate
fragments) later at re-rank time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ingestion.document_loader import RawDocument

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 40  # drop near-empty / noise fragments


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_document(
    doc: RawDocument,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(doc.text)

    chunks: List[Chunk] = []
    for i, text in enumerate(raw_chunks):
        cleaned = " ".join(text.split())
        if len(cleaned) < MIN_CHUNK_CHARS:
            continue  # filter obvious noise (headers, stray punctuation, etc.)
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::chunk_{i}",
                doc_id=doc.doc_id,
                text=cleaned,
                metadata={**doc.metadata, "source_path": doc.source_path, "chunk_index": i},
            )
        )
    return chunks


def chunks_to_langchain_documents(chunks: List[Chunk]) -> List[Document]:
    return [
        Document(page_content=c.text, metadata={**c.metadata, "chunk_id": c.chunk_id, "doc_id": c.doc_id})
        for c in chunks
    ]
