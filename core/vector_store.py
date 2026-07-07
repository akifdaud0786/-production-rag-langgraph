"""
Vector store abstraction.

VECTOR_BACKEND=qdrant  -> local/dev Qdrant or Qdrant Cloud
VECTOR_BACKEND=vertex  -> Vertex AI Vector Search (GCP production)

This lets the same LangGraph retriever code run unchanged in dev and prod,
matching the "Knowledge Store / Vector Store (Vertex AI Vector Search)" box
in the reference architecture while staying cheap and fast to iterate on
locally with Qdrant.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import List

from ingestion.embedding import embed_query, embed_texts

VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "qdrant")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_documents")


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    metadata: dict


class VectorStoreClient:
    """Thin common interface over Qdrant / Vertex AI Vector Search."""

    def upsert(self, ids: List[str], texts: List[str], metadatas: List[dict]) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        raise NotImplementedError


class QdrantStore(VectorStoreClient):
    def __init__(self) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        
        # Robust fallback: if QDRANT_URL is localhost but Qdrant server is not running, 
        # fall back to local disk-based Qdrant client (using SQLite/file database)
        if url.startswith("http://localhost:") or url.startswith("http://127.0.0.1:"):
            try:
                # Test connection with a very short timeout
                test_client = QdrantClient(url=url, api_key=api_key, timeout=1.0)
                test_client.get_collections()
                self.client = test_client
            except Exception:
                # Local service not running, fallback to SQLite-based local storage
                self.client = QdrantClient(path="qdrant_db")
        else:
            self.client = QdrantClient(url=url, api_key=api_key)
            
        self.collection = QDRANT_COLLECTION

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )

    def upsert(self, ids: List[str], texts: List[str], metadatas: List[dict]) -> None:
        from qdrant_client.models import PointStruct

        vectors = embed_texts(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, ids[i])),
                vector=vectors[i],
                payload={"text": texts[i], "chunk_id": ids[i], **metadatas[i]},
            )
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        vector = embed_query(query)
        hits = self.client.search(collection_name=self.collection, query_vector=vector, limit=top_k)
        return [
            RetrievedChunk(
                chunk_id=hit.payload.get("chunk_id", str(hit.id)),
                text=hit.payload.get("text", ""),
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k not in ("text", "chunk_id")},
            )
            for hit in hits
        ]


class VertexVectorSearchStore(VectorStoreClient):
    """Production backend: Vertex AI Vector Search + Firestore for payload/metadata.

    Vertex AI Vector Search only stores vectors + ids efficiently; we keep the
    actual chunk text and metadata in Firestore, matching the "Metadata Store
    (Firestore)" box in the architecture diagram.
    """

    def __init__(self) -> None:
        from google.cloud import aiplatform
        from google.cloud import firestore

        project = os.environ["GCP_PROJECT_ID"]
        region = os.getenv("GCP_REGION", "us-central1")
        aiplatform.init(project=project, location=region)

        self.index_endpoint_id = os.environ["VERTEX_INDEX_ENDPOINT_ID"]
        self.deployed_index_id = os.environ["VERTEX_DEPLOYED_INDEX_ID"]
        self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint(self.index_endpoint_id)
        self.fs = firestore.Client(project=project)
        self.fs_collection = os.getenv("FIRESTORE_COLLECTION", "document_metadata")

    def upsert(self, ids: List[str], texts: List[str], metadatas: List[dict]) -> None:
        from google.cloud import aiplatform

        vectors = embed_texts(texts)
        index = aiplatform.MatchingEngineIndex(os.environ["VERTEX_INDEX_ENDPOINT_ID"])
        datapoints = [
            {"datapoint_id": ids[i], "feature_vector": vectors[i]} for i in range(len(ids))
        ]
        index.upsert_datapoints(datapoints=datapoints)

        batch = self.fs.batch()
        for i, chunk_id in enumerate(ids):
            doc_ref = self.fs.collection(self.fs_collection).document(chunk_id)
            batch.set(doc_ref, {"text": texts[i], **metadatas[i]})
        batch.commit()

    def search(self, query: str, top_k: int = 10) -> List[RetrievedChunk]:
        vector = embed_query(query)
        response = self.index_endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[vector],
            num_neighbors=top_k,
        )
        results: List[RetrievedChunk] = []
        for neighbor in response[0]:
            doc = self.fs.collection(self.fs_collection).document(neighbor.id).get()
            payload = doc.to_dict() or {}
            results.append(
                RetrievedChunk(
                    chunk_id=neighbor.id,
                    text=payload.get("text", ""),
                    score=1.0 - neighbor.distance,  # convert distance -> similarity
                    metadata={k: v for k, v in payload.items() if k != "text"},
                )
            )
        return results


def get_vector_store() -> VectorStoreClient:
    if VECTOR_BACKEND == "vertex":
        return VertexVectorSearchStore()
    return QdrantStore()
