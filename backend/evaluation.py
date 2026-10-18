"""Dataset-based RAG evaluation using an LLM judge."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate

from .workflow import get_llm, workflow

METRIC_NAMES = (
    "context_relevance",
    "faithfulness",
    "answer_relevance",
    "answer_correctness",
)

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict evaluator of a retrieval-augmented assistant. "
            "Score each requested metric from 0.0 to 1.0. "
            "context_relevance measures whether the retrieved context is useful "
            "for answering the question. faithfulness measures whether the answer "
            "is supported by that context; unsupported claims lower the score. "
            "answer_relevance measures whether the answer addresses the question. "
            "answer_correctness compares the answer with the reference answer and "
            "is only requested when a reference is supplied. Do not reward an "
            "answer for unsupported details. Return only a JSON object with "
            "'scores' and 'reasons' objects, keyed by the requested metric names.",
        ),
        (
            "human",
            "Question:\n{question}\n\n"
            "Retrieved context:\n{contexts}\n\n"
            "Assistant answer:\n{answer}\n\n"
            "Reference answer (if supplied):\n{reference_answer}\n\n"
            "Requested metrics: {metrics}\n"
            "Return JSON like "
            '{{"scores":{{"context_relevance":0.0}},'
            '"reasons":{{"context_relevance":"brief reason"}}}}.',
        ),
    ]
)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    raise ValueError("The LLM judge returned content in an unsupported format.")


def parse_judge_response(
    response_text: str, required_metrics: List[str]
) -> Dict[str, Any]:
    """Parse and validate the judge's JSON score response."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("The LLM judge did not return valid JSON.") from exc

    if not isinstance(result, dict) or not isinstance(result.get("scores"), dict):
        raise ValueError("The LLM judge response must contain a scores object.")
    reasons = result.get("reasons", {})
    if not isinstance(reasons, dict):
        raise ValueError("The LLM judge reasons value must be an object.")

    scores: Dict[str, float] = {}
    for metric in required_metrics:
        score = result["scores"].get(metric)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"The LLM judge omitted a numeric {metric} score.")
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"The {metric} score must be between 0.0 and 1.0.")
        scores[metric] = float(score)

    return {
        "scores": scores,
        "reasons": {metric: str(reasons.get(metric, "")) for metric in required_metrics},
    }


def evaluate_example(
    example: Dict[str, Any], judge: Optional[Any] = None
) -> Dict[str, Any]:
    """Run the RAG workflow and score one dataset example."""
    question = example.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Each evaluation example must have a non-empty question.")

    result = workflow.invoke({"query": question.strip(), "use_memory": False})
    answer = result.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("The RAG workflow did not return a non-empty answer.")

    contexts = [
        document.page_content for document in result.get("retrieved_docs", [])
    ]
    reference_answer = example.get("reference_answer")
    if reference_answer is not None and not isinstance(reference_answer, str):
        raise ValueError("reference_answer must be a string when provided.")
    required_metrics = list(METRIC_NAMES[:3])
    if reference_answer:
        required_metrics.append("answer_correctness")

    formatted = JUDGE_PROMPT.format_messages(
        question=question,
        contexts="\n---\n".join(contexts) or "No context retrieved.",
        answer=answer,
        reference_answer=reference_answer or "Not supplied.",
        metrics=", ".join(required_metrics),
    )
    judge_response = (judge or get_llm()).invoke(formatted)
    evaluation = parse_judge_response(
        _message_text(judge_response.content), required_metrics
    )

    return {
        "question": question,
        "answer": answer,
        "reference_answer": reference_answer,
        "contexts": contexts,
        **evaluation,
    }


def aggregate_scores(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average each metric over examples that were scored for that metric."""
    totals: Dict[str, List[float]] = {}
    for result in results:
        for metric, score in result["scores"].items():
            totals.setdefault(metric, []).append(score)
    return {
        metric: sum(scores) / len(scores)
        for metric, scores in totals.items()
        if scores
    }


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    """Load a JSON Lines evaluation dataset."""
    examples = []
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
            examples.append(example)
    if not examples:
        raise ValueError("The evaluation dataset contains no examples.")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG workflow with an LLM judge."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="JSONL file with question and optional reference_answer fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Path for the JSON report (default: evaluation_results.json).",
    )
    args = parser.parse_args()

    results = [
        evaluate_example(example) for example in load_dataset(args.dataset)
    ]
    report = {"summary": aggregate_scores(results), "examples": results}
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))
    print(f"Evaluation report written to {args.output}")


if __name__ == "__main__":
    main()
