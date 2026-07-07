"""
RAGAS evaluation harness — the "RAGAS evaluation suite" box in the
architecture diagram (Golden dataset · 15 samples · 6 tests, F/R/P/C metrics,
Judge LLM).

Usage:
    python -m evaluation.ragas_eval
    python -m evaluation.ragas_eval --output results.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluation.ragas_eval")

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def _run_pipeline_for_eval(question: str) -> dict:
    """Runs the real LangGraph pipeline and extracts what RAGAS needs."""
    import requests
    
    # Try calling the running FastAPI server first to avoid Qdrant local file lock conflicts
    api_url = os.getenv("API_URL", "http://localhost:8000")
    try:
        response = requests.post(
            f"{api_url}/query",
            json={"query": question, "thread_id": f"eval-{abs(hash(question))}"},
            timeout=180
        )
        if response.status_code == 200:
            data = response.json()
            contexts = [s["text_preview"] for s in data.get("sources", [])]
            return {
                "answer": data.get("answer", ""),
                "contexts": contexts or ["(no context retrieved)"],
            }
        else:
            raise RuntimeError(f"API server returned status {response.status_code}: {response.text}")
    except (requests.exceptions.ConnectionError, requests.exceptions.ConnectTimeout):
        logger.warning("API server connection refused. Falling back to direct local execution.")
    except Exception as exc:
        logger.error("API query failed during evaluation: %s", exc)
        raise

    # Fallback to direct Python import/execution if API server is not running
    from core.graph import run_query

    state = run_query(question, thread_id=f"eval-{abs(hash(question))}")
    contexts = [c["text"] for c in state.get("reranked_chunks", [])]
    return {
        "answer": state.get("final_answer", ""),
        "contexts": contexts or ["(no context retrieved)"],
    }


def build_evaluation_dataset(golden_samples: List[dict]):
    """Runs the pipeline for every golden question and assembles a RAGAS-ready dataset."""
    from datasets import Dataset

    questions, answers, contexts, ground_truths = [], [], [], []

    for sample in golden_samples:
        logger.info("Running pipeline for eval question: %s", sample["question"])
        result = _run_pipeline_for_eval(sample["question"])
        questions.append(sample["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(sample["ground_truth"])

    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )


def run_evaluation(golden_dataset_path: Path = GOLDEN_DATASET_PATH) -> dict:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    golden_samples = json.loads(golden_dataset_path.read_text())
    dataset = build_evaluation_dataset(golden_samples)

    from langchain_groq import ChatGroq
    from langchain_huggingface import HuggingFaceEmbeddings

    groq_llm = ChatGroq(
        model=os.getenv("GROQ_PRIMARY_MODEL", "llama-3.3-70b-versatile"),
        api_key=os.getenv("GROQ_API_KEY"),
    )
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )

    from ragas.run_config import RunConfig

    rate_friendly_config = RunConfig(
        max_workers=2,
        max_retries=20,
        timeout=180
    )

    logger.info("Running RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=groq_llm,
        embeddings=embeddings,
        run_config=rate_friendly_config,
    )

    scores = result.to_pandas().mean(numeric_only=True).to_dict()
    logger.info("RAGAS results: %s", scores)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation against the golden dataset.")
    parser.add_argument("--dataset", type=str, default=str(GOLDEN_DATASET_PATH))
    parser.add_argument("--output", type=str, default="evaluation/results.json")
    args = parser.parse_args()

    scores = run_evaluation(Path(args.dataset))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(scores, indent=2))
    logger.info("Wrote evaluation results to %s", output_path)


if __name__ == "__main__":
    main()
