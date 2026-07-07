# Production-Grade Advanced RAG — LangGraph · GCP · Groq

An enterprise-grade, cyclic Retrieval-Augmented Generation system that tells the
difference between real technical "true data" and irrelevant "noisy data" using
history-aware planning, semantic re-ranking, and a self-critique/refine loop —
wrapped in an LLM gateway, guardrails, and a full RAGAS evaluation suite.

This matches the two reference architecture diagrams:
1. **Cyclic RAG Workflow (LangGraph)** — Query Understanding → Retrieve → Generate
   → Critique & Evaluate → (loop: Refine Query / Retrieve Again) → Final Answer.
2. **System Architecture** — Interface layer, API + Safety layer, LangGraph
   agentic core, Retrieval layer, LLM gateway, Ingestion pipeline, Observability,
   RAGAS evaluation suite, GCP infrastructure, Terraform IaC.

---

## 1. High-level architecture

```
                        ┌─────────────────────────────┐
                        │   Streamlit Chat UI / Eval  │
                        └───────────────┬─────────────┘
                                        │
                        ┌───────────────▼─────────────┐
                        │ FastAPI /query  + Guardrails │
                        └───────────────┬─────────────┘
                                        │
        ┌───────────────────────────────▼───────────────────────────────┐
        │                    LangGraph Agentic Core                     │
        │  Query Understanding → Retrieve → Generate → Critique/Eval    │
        │              ▲                                   │ not good  │
        │              └───────────── Refine Query ─────────┘          │
        │                              │ good enough                    │
        │                         Final Answer                          │
        └───────────────────────────────┬───────────────────────────────┘
                                        │
        ┌───────────────────────────────▼───────────────────────────────┐
        │  Retrieval layer: Vector DB (Qdrant/Vertex AI Vector Search)   │
        │  + FlashRank local re-ranker for true-data vs noisy-data       │
        └───────────────────────────────┬───────────────────────────────┘
                                        │
        ┌───────────────────────────────▼───────────────────────────────┐
        │   LLM Gateway: Groq (Llama 3.3 70B primary, 3.1 8B fallback)   │
        └─────────────────────────────────────────────────────────────┘
```

## 2. Repo layout

```
production-rag-langgraph/
├── ingestion/          # Document loading, chunking, embedding, ingestion pipeline
├── core/               # LangGraph nodes + graph definition (the agentic core)
├── gateway/            # Unified LLM gateway with primary/fallback + retries
├── guardrails/         # Input/output guardrails (PII, prompt-injection, topical)
├── evaluation/         # RAGAS golden dataset + evaluation harness
├── api/                # FastAPI service (REST entrypoint)
├── frontend/           # Streamlit chat UI + evaluation dashboard
├── observability/      # Structured logging / tracing configuration
├── terraform/          # GCP infrastructure as code
├── tests/              # Unit + integration tests
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yml  # Local dev: Qdrant + API + Streamlit
├── cloudbuild.yaml     # CI/CD pipeline definition for Cloud Build
└── requirements.txt
```

## 3. Quick start (local)

```bash
cp .env.example .env          # fill in GROQ_API_KEY etc.
docker compose up --build     # starts Qdrant, API (localhost:8000), UI (localhost:8501)
```

Or run natively:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Start local Qdrant (or point QDRANT_URL at Qdrant Cloud)
docker run -p 6333:6333 qdrant/qdrant

# 2. Ingest documents
python -m ingestion.pipeline --source ./data/docs

# 3. Run the API
uvicorn api.main:app --reload --port 8000

# 4. Run the chat UI
streamlit run frontend/streamlit_app.py

# 5. Run evaluations
python -m evaluation.ragas_eval
```

## 4. Cloud deployment (GCP)

```bash
cd terraform
terraform init
terraform apply -var="project_id=YOUR_GCP_PROJECT" -var="groq_api_key=YOUR_KEY"
```

This provisions: VPC + connector, GCS buckets (raw + processed), Artifact
Registry, Cloud Run services (API + Streamlit), and IAM bindings. Vertex AI
Vector Search (the production vector store) is provisioned via the
`vertex_vector_search` module — swap this in for local Qdrant once you move
past prototyping.

CI/CD: `cloudbuild.yaml` builds container images, pushes to Artifact Registry,
and deploys to Cloud Run on every push to `main`.

## 5. Why "true data" vs "noisy data"

Real corpora mix authoritative technical content with boilerplate, marketing
copy, changelogs, or irrelevant tangents. Two mechanisms handle this:

- **Semantic re-ranking** (`core/retriever.py`): FlashRank cross-encoder scores
  each retrieved chunk against the *reformulated* query, not the raw one, and
  chunks below a relevance threshold are dropped before generation.
- **Critique & Evaluate node** (`core/critique.py`): after generation, a
  lightweight LLM judge checks whether the answer is actually grounded in the
  retrieved "true data" chunks. If not, the graph loops back, rewrites the
  query (`core/planner.py`'s `refine_query`), and retrieves again — up to
  `MAX_REFINE_ITERATIONS`.

## 6. Key design decisions

| Concern | Choice | Rationale |
|---|---|---|
| Orchestration | LangGraph `StateGraph` w/ `MemorySaver` checkpointer | native cycles + conversation memory |
| Vector store | Qdrant (local/dev) → Vertex AI Vector Search (prod) | swappable via `VECTOR_BACKEND` env var |
| Re-ranker | FlashRank (local, no extra API cost) | fast cross-encoder re-ranking |
| LLM inference | Groq (Llama 3.3 70B primary / Llama 3.1 8B fallback) | ultra-low latency + high throughput |
| Gateway | Custom unified gateway (Portkey-compatible interface) | retries, fallback, cost/latency logging |
| Guardrails | NeMo-Guardrails-style config + custom filters | topical rails, PII redaction, jailbreak defense |
| Evaluation | RAGAS (faithfulness, answer relevancy, context precision/recall) | industry-standard RAG metrics |
| IaC | Terraform, modularized | reusable, reviewable, environment-parameterized |

## 7. Environment variables

See `.env.example` for the full list. Minimum to run locally: `GROQ_API_KEY`,
`QDRANT_URL`, `QDRANT_API_KEY` (if using Qdrant Cloud).
