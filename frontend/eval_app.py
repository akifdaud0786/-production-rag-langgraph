"""
Streamlit Eval App — the "Interface layer / Streamlit Eval App / Evaluation UI" box.

Displays RAGAS results (faithfulness, answer relevancy, context precision/recall)
from evaluation/results.json, and lets you trigger a fresh evaluation run.

Run: streamlit run frontend/eval_app.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

RESULTS_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "results.json"
GOLDEN_DATASET_PATH = Path(__file__).resolve().parent.parent / "evaluation" / "golden_dataset.json"

st.set_page_config(page_title="RAG Evaluation Dashboard", page_icon="📊", layout="centered")
st.title("📊 RAGAS Evaluation Dashboard")
st.caption("Faithfulness · Answer Relevancy · Context Precision · Context Recall")

golden = json.loads(GOLDEN_DATASET_PATH.read_text()) if GOLDEN_DATASET_PATH.exists() else []
st.metric("Golden dataset size", len(golden))

if st.button("Run evaluation now (this calls the live LLM gateway)"):
    with st.spinner("Running pipeline + RAGAS metrics across the golden dataset…"):
        proc = subprocess.run(
            [sys.executable, "-m", "evaluation.ragas_eval"],
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            st.error(proc.stderr[-3000:])
        else:
            st.success("Evaluation complete.")
            st.rerun()

if RESULTS_PATH.exists():
    scores = json.loads(RESULTS_PATH.read_text())
    labels = list(scores.keys())
    values = [round(v, 3) for v in scores.values()]

    fig = go.Figure(data=[go.Bar(x=labels, y=values, marker_color="#4F46E5")])
    fig.update_layout(yaxis_range=[0, 1], title="RAGAS metric scores (0–1)")
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(len(labels))
    for col, label, value in zip(cols, labels, values):
        col.metric(label, value)
else:
    st.info("No evaluation results yet — click the button above or run `python -m evaluation.ragas_eval`.")
