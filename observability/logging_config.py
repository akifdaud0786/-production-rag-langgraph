"""
Observability layer — matches "Pydantic Logfire / Span tracing" and
"LangSmith / Agent traces" in the architecture diagram, plus Cloud Logging /
Cloud Monitoring / Cloud Trace / Alerts in the GCP box.

- Structured console + Cloud Logging handler (auto-detected when running on GCP).
- Optional Logfire instrumentation for span-level tracing of the LangGraph run.
- Optional LangSmith tracing (enable via LANGCHAIN_TRACING_V2=true).
"""
from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Google Cloud Logging — active automatically on Cloud Run, no-op locally if
    # credentials aren't available.
    if os.getenv("GCP_PROJECT_ID"):
        try:
            import google.cloud.logging as gcp_logging

            client = gcp_logging.Client()
            client.setup_logging(log_level=getattr(logging, level, logging.INFO))
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Cloud Logging not configured: %s", exc)

    # Pydantic Logfire — span/trace instrumentation for the LangGraph pipeline.
    logfire_token = os.getenv("LOGFIRE_TOKEN")
    if logfire_token:
        try:
            import logfire

            logfire.configure(token=logfire_token)
            logfire.instrument_httpx()
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("Logfire not configured: %s", exc)

    # LangSmith tracing for the agentic core (set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY).
    if os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true":
        os.environ.setdefault("LANGCHAIN_PROJECT", "production-rag-langgraph")
        logging.getLogger(__name__).info("LangSmith tracing enabled for project=%s", os.environ["LANGCHAIN_PROJECT"])
