"""
FastAPI entrypoint — the "API + Safety layer" in the architecture diagram.

    POST /query   -> runs guardrails, invokes the LangGraph agentic core,
                     runs output guardrails, returns the final answer + sources.
    GET  /health  -> liveness/readiness probe for Cloud Run.
"""
from __future__ import annotations

import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import shutil

from api.schemas import ChunkResponse, HealthResponse, QueryRequest, QueryResponse
from core.graph import run_query
from guardrails.filters import run_input_guardrails, run_output_guardrails
from observability.logging_config import configure_logging

load_dotenv()
configure_logging()
logger = logging.getLogger("api.main")

app = FastAPI(
    title="Production-Grade Advanced RAG API",
    description="Cyclic LangGraph RAG system with guardrails, LLM gateway, and evaluation support.",
    version="1.0.0",
)

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("Pre-warming model dependencies (embeddings and reranker)...")
    try:
        from ingestion.embedding import get_embedder
        from core.retriever import _get_reranker
        get_embedder()
        _get_reranker()
        logger.info("Model pre-warming complete. Ready to serve requests!")
    except Exception as e:
        logger.warning("Failed to pre-warm models: %s", e)


GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", vector_backend=os.getenv("VECTOR_BACKEND", "qdrant"))


@app.post("/ingest")
def ingest(file: UploadFile = File(...)):
    logger.info("Ingestion request received for file: %s", file.filename)
    temp_dir = Path("data/temp_upload")
    temp_dir.mkdir(parents=True, exist_ok=True)
    file_path = temp_dir / file.filename
    
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        from ingestion.pipeline import run as run_ingestion
        run_ingestion(source_dir=str(temp_dir), gcs_bucket=None, gcs_prefix="")
        return {"status": "success", "message": f"Successfully ingested {file.filename}"}
    except Exception as e:
        logger.exception("Ingestion failed for %s", file.filename)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if file_path.exists():
            file_path.unlink()
        try:
            temp_dir.rmdir()
        except Exception:
            pass



@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    start = time.perf_counter()

    if GUARDRAILS_ENABLED:
        input_check = run_input_guardrails(request.query)
        if not input_check.allowed:
            logger.warning("Blocked request thread_id=%s reason=%s", request.thread_id, input_check.reason)
            raise HTTPException(status_code=400, detail=f"Request blocked by guardrails: {input_check.reason}")
        sanitized_query = input_check.sanitized_text
    else:
        sanitized_query = request.query

    try:
        state = run_query(sanitized_query, thread_id=request.thread_id)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Graph execution failed for thread_id=%s", request.thread_id)
        raise HTTPException(status_code=502, detail=f"RAG pipeline error: {exc}") from exc

    answer = state.get("final_answer", "")
    if GUARDRAILS_ENABLED:
        output_check = run_output_guardrails(answer)
        answer = output_check.sanitized_text

    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "query_served thread_id=%s refine_count=%d satisfactory=%s latency_ms=%.1f",
        request.thread_id, state.get("refine_count", 0), state.get("is_satisfactory"), elapsed_ms,
    )

    sources = [
        ChunkResponse(chunk_id=c["chunk_id"], score=c["score"], text_preview=c["text"][:200])
        for c in state.get("reranked_chunks", [])
    ]

    return QueryResponse(
        answer=answer,
        standalone_query=state.get("standalone_query", sanitized_query),
        refine_count=state.get("refine_count", 0),
        is_satisfactory=state.get("is_satisfactory", True),
        sources=sources,
        trace=state.get("trace"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=os.getenv("API_HOST", "0.0.0.0"), port=int(os.getenv("API_PORT", "8000")))
