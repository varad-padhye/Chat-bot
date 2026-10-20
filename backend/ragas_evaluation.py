"""Evaluate the chatbot with Ragas metrics using a JSON Lines test dataset."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset
from langchain_core.embeddings import Embeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

from .embeddings import get_embedding_function
from .workflow import get_llm, workflow


class ChatbotEmbeddings(Embeddings):
    """Adapt the project's single-text embedder to LangChain's batch API."""

    def __init__(self, embed_query: Callable[[str], List[float]]) -> None:
        self._embed_query = embed_query

    def embed_query(self, text: str) -> List[float]:
        return self._embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]


BASE_METRICS = (faithfulness, answer_relevancy)
REFERENCE_METRICS = (context_precision, context_recall, answer_correctness)


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load and validate JSONL rows with question and optional reference answer."""
    examples: List[Dict[str, Any]] = []
    with dataset_path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on dataset line {line_number}."
                ) from exc
            if not isinstance(example, dict):
                raise ValueError(
                    f"Dataset line {line_number} must contain a JSON object."
                )
            question = example.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(
                    f"Dataset line {line_number} needs a non-empty question."
                )
            reference = example.get(
                "reference_answer", example.get("ground_truth")
            )
            if reference is not None and (
                not isinstance(reference, str) or not reference.strip()
            ):
                raise ValueError(
                    f"Dataset line {line_number} has an invalid reference answer."
                )
            examples.append(
                {
                    "question": question.strip(),
                    "reference_answer": reference.strip() if reference else None,
                }
            )
    if not examples:
        raise ValueError("The evaluation dataset contains no examples.")
    return examples


def _format_context(document: Any) -> str:
    metadata = getattr(document, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    source = metadata.get("source")
    if not isinstance(source, str) or not source.strip():
        source = "Unknown source"
    page = metadata.get("page")
    if isinstance(page, int) and not isinstance(page, bool) and page > 0:
        source = f"{source}, page {page}"
    content = getattr(document, "page_content", "")
    return f"[Source: {source}]\n{content}"


def collect_rag_outputs(
    examples: List[Dict[str, Any]], workflow_runner: Any = workflow
) -> List[Dict[str, Any]]:
    """Run test questions through RAG without user-specific long-term memory."""
    rows = []
    for index, example in enumerate(examples, start=1):
        result = workflow_runner.invoke(
            {"query": example["question"], "use_memory": False}
        )
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(
                f"The RAG workflow returned no answer for dataset row {index}."
            )
        documents = result.get("retrieved_docs", [])
        if not isinstance(documents, list):
            raise ValueError(
                f"The RAG workflow returned invalid contexts for dataset row {index}."
            )
        rows.append(
            {
                "question": example["question"],
                "answer": answer.strip(),
                "contexts": [_format_context(document) for document in documents],
                "reference_answer": example.get("reference_answer"),
            }
        )
    return rows


def _ragas_rows(
    rows: List[Dict[str, Any]], include_reference: bool
) -> List[Dict[str, Any]]:
    dataset_rows = []
    for row in rows:
        dataset_row = {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": row["contexts"],
        }
        if include_reference:
            dataset_row["reference"] = row["reference_answer"]
        dataset_rows.append(dataset_row)
    return dataset_rows


def _validated_scores(result: Any, metric_names: List[str]) -> List[Dict[str, float]]:
    scores = getattr(result, "scores", None)
    if not isinstance(scores, list):
        raise ValueError("Ragas did not return per-example scores.")
    if len(scores) == 0:
        return []

    validated: List[Dict[str, float]] = []
    for index, score_row in enumerate(scores, start=1):
        if not isinstance(score_row, dict):
            raise ValueError(f"Ragas returned invalid scores for row {index}.")
        row_scores: Dict[str, float] = {}
        for metric_name in metric_names:
            score = score_row.get(metric_name)
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"Ragas did not return a numeric {metric_name} score "
                    f"for row {index}."
                )
            if not math.isfinite(score):
                raise ValueError(
                    f"Ragas returned a non-finite {metric_name} score for row {index}."
                )
            row_scores[metric_name] = float(score)
        validated.append(row_scores)
    return validated


def evaluate_dataset(
    examples: List[Dict[str, Any]],
    workflow_runner: Any = workflow,
    ragas_evaluator: Callable[..., Any] = evaluate,
    llm: Optional[Any] = None,
    embeddings: Optional[Embeddings] = None,
) -> Dict[str, Any]:
    """Score RAG outputs with faithfulness and answer relevance, plus references."""
    rows = collect_rag_outputs(examples, workflow_runner=workflow_runner)
    reference_indices = [
        index
        for index, row in enumerate(rows)
        if row["reference_answer"] is not None
    ]

    configured_llm = LangchainLLMWrapper(llm or get_llm())
    configured_embeddings = LangchainEmbeddingsWrapper(
        embeddings
        or ChatbotEmbeddings(get_embedding_function())
    )

    base_result = ragas_evaluator(
        dataset=Dataset.from_list(_ragas_rows(rows, include_reference=False)),
        metrics=list(BASE_METRICS),
        llm=configured_llm,
        embeddings=configured_embeddings,
        raise_exceptions=True,
        show_progress=False,
    )
    base_names = [metric.name for metric in BASE_METRICS]
    base_scores = _validated_scores(base_result, base_names)
    if len(base_scores) != len(rows):
        raise ValueError("Ragas returned a different number of base scores than inputs.")
    for row, scores in zip(rows, base_scores):
        row["scores"] = scores

    reference_metric_names: List[str] = []
    if reference_indices:
        reference_examples = [rows[index] for index in reference_indices]
        reference_result = ragas_evaluator(
            dataset=Dataset.from_list(
                _ragas_rows(reference_examples, include_reference=True)
            ),
            metrics=list(REFERENCE_METRICS),
            llm=configured_llm,
            embeddings=configured_embeddings,
            raise_exceptions=True,
            show_progress=False,
        )
        reference_metric_names = [
            metric.name for metric in REFERENCE_METRICS
        ]
        reference_scores = _validated_scores(
            reference_result, reference_metric_names
        )
        if len(reference_scores) != len(reference_indices):
            raise ValueError(
                "Ragas returned a different number of reference scores than inputs."
            )
        for index, scores in zip(reference_indices, reference_scores):
            rows[index]["scores"].update(scores)

    all_metric_names = base_names + reference_metric_names
    summary = {
        metric_name: sum(row["scores"][metric_name] for row in rows if metric_name in row["scores"])
        / sum(metric_name in row["scores"] for row in rows)
        for metric_name in all_metric_names
    }
    return {"summary": summary, "examples": rows}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG chatbot with Ragas."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL dataset containing question and optional reference_answer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ragas_evaluation_results.json"),
        help="Path to the JSON report (default: ragas_evaluation_results.json).",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    from .database import init_db

    init_db()
    report = evaluate_dataset(load_dataset(args.dataset))
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Ragas evaluation report written to {args.output}")


if __name__ == "__main__":
    main()
