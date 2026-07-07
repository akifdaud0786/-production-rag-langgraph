"""
Production-Grade RAG Console — Combined Chat UI & Evaluation Dashboard.

Run: streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st

# Configure Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
RESULTS_PATH = ROOT_DIR / "evaluation" / "results.json"
GOLDEN_DATASET_PATH = ROOT_DIR / "evaluation" / "golden_dataset.json"
API_URL = os.getenv("API_URL", "http://localhost:8000")

# 1. Page Configuration
st.set_page_config(
    page_title="Enterprise RAG Console",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Custom Sleek CSS for Dark-Mode/Glassmorphic aesthetics
st.markdown(
    """
    <style>
    /* Main container background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.9) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 600;
        color: #94a3b8;
        padding: 12px 16px;
        border-radius: 4px 4px 0px 0px;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
    }
    .stTabs [aria-selected="true"] {
        color: #818cf8 !important;
        border-bottom: 2px solid #818cf8 !important;
    }
    
    /* Custom Card/Box for Stats & Results */
    .metric-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(12px);
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sleek Title and Caption */
    .header-title {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    /* Scrollbar customization */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a;
    }
    ::-webkit-scrollbar-thumb {
        background: #334155;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #475569;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 3. Sidebar Configuration (Session & API Info)
with st.sidebar:
    st.image(
        "https://img.icons8.com/external-flat-icons-inspirational-tuts/100/external-brain-mind-and-mental-flat-icons-inspirational-tuts.png",
        width=70,
    )
    st.markdown("### **Enterprise RAG**")
    st.caption("Active Console Config")
    st.divider()

    # Session Thread ID Management
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []

    st.subheader("Session Control")
    st.text_input("Active Thread ID", value=st.session_state.thread_id, disabled=True)
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.history = []
        st.rerun()

    st.divider()
    st.subheader("Gateway API URL")
    st.code(API_URL, language="bash")

    st.divider()
    st.subheader("📁 Upload Document")
    st.caption("Ingest new files directly into the Qdrant database.")
    
    uploaded_file = st.file_uploader(
        "Choose a file", 
        type=["txt", "pdf", "docx"], 
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        temp_dir = ROOT_DIR / "data" / "temp_upload"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = temp_dir / uploaded_file.name
        
        # Save the uploaded file locally
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Parsing and vectorizing doc..."):
            try:
                # Lazy load ingestion pipeline
                from ingestion.pipeline import run as run_ingestion
                
                # Run the pipeline locally
                run_ingestion(source_dir=str(temp_dir), gcs_bucket=None, gcs_prefix="")
                st.success(f"🎉 Successfully Ingested: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
            finally:
                # Clean up file and directory
                if file_path.exists():
                    file_path.unlink()
                try:
                    temp_dir.rmdir()
                except Exception:
                    pass

# 4. Main Console Header
st.markdown("<h1 class='header-title'>🧠 Enterprise RAG Console</h1>", unsafe_allow_html=True)
st.caption(
    "Powered by LangGraph Cyclic Core · Groq Llama 3.3 · Programmatic Guardrails · RAGAS Benchmarking"
)
st.spacer = st.empty()

# 5. Define Navigation Tabs
tab1, tab2 = st.tabs(["💬 RAG Chat Assistant", "📊 RAGAS Analytics Dashboard"])

# ---------------------------------------------------------------------------
# TAB 1: Chat Assistant
# ---------------------------------------------------------------------------
with tab1:
    st.markdown("### **Interactive Assistant**")
    st.caption("Chat with the knowledge graph. Prompt-injection checks & PII redaction are active.")
    st.write("")

    # Display Conversation History
    for turn in st.session_state.history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander(
                    f"🔍 View Sources ({len(turn['sources'])}) · refine_count={turn.get('refine_count', 0)}"
                ):
                    for s in turn["sources"]:
                        st.markdown(
                            f"📌 **{s['chunk_id']}** (Similarity Score: `{s['score']:.2f}`)\n\n> {s['text_preview']}"
                        )

    # Chat Input
    if prompt := st.chat_input("Ask a technical question about the infrastructure or architecture..."):
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Executing LangGraph (retrieving, evaluating, and refining)..."):
                try:
                    response = requests.post(
                        f"{API_URL}/query",
                        json={"query": prompt, "thread_id": st.session_state.thread_id},
                        timeout=150,
                    )
                    response.raise_for_status()
                    data = response.json()
                    answer = data["answer"]
                    st.markdown(answer)

                    if not data["is_satisfactory"]:
                        st.warning("⚠️ Note: Maximum query refinement iterations reached. Answer may be incomplete.")

                    if data["sources"]:
                        with st.expander(
                            f"🔍 View Sources ({len(data['sources'])}) · refine_count={data['refine_count']}"
                        ):
                            for s in data["sources"]:
                                st.markdown(
                                    f"📌 **{s['chunk_id']}** (Similarity Score: `{s['score']:.2f}`)\n\n> {s['text_preview']}"
                                )

                    # Save to history
                    st.session_state.history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": data["sources"],
                            "refine_count": data["refine_count"],
                        }
                    )
                except requests.exceptions.RequestException as exc:
                    error_msg = f"❌ **Gateway Query Failed:** {exc}"
                    st.error(error_msg)
                    st.session_state.history.append({"role": "assistant", "content": error_msg})

# ---------------------------------------------------------------------------
# TAB 2: Evaluation Dashboard
# ---------------------------------------------------------------------------
with tab2:
    st.markdown("### **RAGAS Evaluation Dashboard**")
    st.caption("Benchmark faithfulness, relevancy, and context quality metrics using the Golden Dataset.")
    st.write("")

    # Display Golden Dataset Info
    golden = []
    if GOLDEN_DATASET_PATH.exists():
        try:
            golden = json.loads(GOLDEN_DATASET_PATH.read_text())
        except Exception:
            pass
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value'>{len(golden)}</div>
                <div class='metric-label'>Golden Test Samples</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_stat2:
        status_text = "Ready to Benchmark" if RESULTS_PATH.exists() else "Awaiting Initial Run"
        st.markdown(
            f"""
            <div class='metric-card'>
                <div class='metric-value' style='color:#a78bfa;'>{status_text}</div>
                <div class='metric-label'>Evaluation Status</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    st.write("")
    st.divider()

    # Initialize session state for background evaluation
    if "eval_status" not in st.session_state:
        st.session_state.eval_status = "idle"  # idle, running, completed, failed
    if "eval_error" not in st.session_state:
        st.session_state.eval_error = ""

    def run_eval_async():
        try:
            import subprocess
            import sys
            proc = subprocess.run(
                [sys.executable, "-m", "evaluation.ragas_eval"],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
            )
            if proc.returncode != 0:
                st.session_state.eval_status = "failed"
                st.session_state.eval_error = proc.stderr
            else:
                st.session_state.eval_status = "completed"
        except Exception as e:
            st.session_state.eval_status = "failed"
            st.session_state.eval_error = str(e)

    # Render Evaluation Controls
    if st.session_state.eval_status == "running":
        st.warning("⏳ **RAGAS Evaluation is running in the background...**")
        st.info("This benchmarks the system against 15 reference questions. Since it makes 60+ Groq LLM calls, it will take 3-5 minutes. You can continue using the chat assistant while it runs.")
        if st.button("🔄 Refresh / Check Status", use_container_width=True):
            st.rerun()
    else:
        if st.button("⚡ Run Full RAGAS Evaluation (Queries Groq LLM)", use_container_width=True):
            st.session_state.eval_status = "running"
            st.session_state.eval_error = ""
            import threading
            t = threading.Thread(target=run_eval_async)
            t.start()
            st.rerun()

    if st.session_state.eval_status == "completed":
        st.success("🎉 RAGAS Evaluation complete. Results loaded below.")
        st.session_state.eval_status = "idle"
        st.rerun()
    elif st.session_state.eval_status == "failed":
        st.error("❌ RAGAS Pipeline Execution Failed:")
        st.code(st.session_state.eval_error[-3000:], language="bash")
        st.session_state.eval_status = "idle"

    # Render Charts if Results exist
    if RESULTS_PATH.exists():
        try:
            scores = json.loads(RESULTS_PATH.read_text())
            labels = [l.replace("_", " ").title() for l in scores.keys()]
            values = [round(v, 3) for v in scores.values()]

            # Plotly bar chart
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=labels,
                        y=values,
                        marker_color=["#38bdf8", "#60a5fa", "#818cf8", "#a78bfa"],
                        text=values,
                        textposition="auto",
                    )
                ]
            )
            fig.update_layout(
                yaxis_range=[0, 1],
                title="RAGAS Accuracy Metrics (Scale 0.0 - 1.0)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f8fafc",
                margin=dict(t=50, b=20, l=10, r=10),
            )
            fig.update_yaxes(showgrid=True, gridcolor="rgba(255, 255, 255, 0.05)")
            fig.update_xaxes(showgrid=False)

            st.plotly_chart(fig, use_container_width=True)

            # Display individual cards for metrics
            st.markdown("#### **Metric Breakdown**")
            cols = st.columns(len(scores))
            colors = ["#38bdf8", "#60a5fa", "#818cf8", "#a78bfa"]
            for col, (metric_key, val), color in zip(cols, scores.items(), colors):
                metric_name = metric_key.replace("_", " ").title()
                col.markdown(
                    f"""
                    <div class='metric-card'>
                        <div class='metric-value' style='color:{color};'>{val:.3f}</div>
                        <div class='metric-label'>{metric_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        except Exception as e:
            st.error(f"Failed to read evaluation results: {e}")
    else:
        st.info("💡 No evaluation results found. Click the button above to run RAGAS benchmark evaluation.")
