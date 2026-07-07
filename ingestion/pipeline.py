"""
Ingestion pipeline entrypoint.

    python -m ingestion.pipeline --source ./data/docs
    python -m ingestion.pipeline --gcs-bucket my-raw-bucket --gcs-prefix docs/

Loads documents -> chunks -> embeds -> upserts into the configured vector
store (Qdrant locally, Vertex AI Vector Search in production), and optionally
mirrors cleaned text back into the GCS "processed" bucket.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from ingestion.chunking import chunk_document
from ingestion.document_loader import (
    load_from_gcs,
    load_local_documents,
    write_processed_to_gcs,
)
from core.vector_store import get_vector_store

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ingestion.pipeline")


def run(source_dir: str | None, gcs_bucket: str | None, gcs_prefix: str, batch_size: int = 64) -> None:
    if gcs_bucket:
        documents = load_from_gcs(gcs_bucket, gcs_prefix)
    elif source_dir:
        documents = load_local_documents(source_dir)
    else:
        raise ValueError("Provide either --source or --gcs-bucket")

    store = get_vector_store()
    processed_bucket = os.getenv("GCS_PROCESSED_BUCKET")

    total_chunks = 0
    total_docs = 0
    ids_batch, texts_batch, meta_batch = [], [], []

    for doc in documents:
        total_docs += 1
        chunks = chunk_document(doc)
        logger.info("Loaded %-40s -> %d chunk(s)", doc.metadata.get("filename", doc.doc_id), len(chunks))

        if processed_bucket and gcs_bucket:
            write_processed_to_gcs(processed_bucket, doc)

        for chunk in chunks:
            ids_batch.append(chunk.chunk_id)
            texts_batch.append(chunk.text)
            meta_batch.append(chunk.metadata)
            total_chunks += 1

            if len(ids_batch) >= batch_size:
                store.upsert(ids_batch, texts_batch, meta_batch)
                ids_batch, texts_batch, meta_batch = [], [], []

    if ids_batch:
        store.upsert(ids_batch, texts_batch, meta_batch)

    logger.info("Ingestion complete: %d document(s), %d chunk(s) indexed.", total_docs, total_chunks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("--source", type=str, default=None, help="Local directory of documents.")
    parser.add_argument("--gcs-bucket", type=str, default=None, help="GCS raw bucket name.")
    parser.add_argument("--gcs-prefix", type=str, default="", help="Prefix/folder inside the GCS bucket.")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if not args.source and not args.gcs_bucket:
        logger.error("Must specify --source <dir> or --gcs-bucket <bucket>")
        sys.exit(1)

    run(args.source, args.gcs_bucket, args.gcs_prefix, args.batch_size)


if __name__ == "__main__":
    main()
