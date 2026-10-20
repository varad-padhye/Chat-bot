import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from backend.ragas_evaluation import (
    collect_rag_outputs,
    evaluate_dataset,
    load_dataset,
)


class RagasEvaluationTests(unittest.TestCase):
    def test_load_dataset_accepts_reference_answer_and_ground_truth(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "eval.jsonl"
            dataset_path.write_text(
                json.dumps(
                    {
                        "question": "  What is entropy?  ",
                        "reference_answer": "A measure of disorder.",
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "question": "Explain this topic.",
                        "ground_truth": "Use the document context.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            examples = load_dataset(dataset_path)

        self.assertEqual(examples[0]["question"], "What is entropy?")
        self.assertEqual(
            examples[1]["reference_answer"], "Use the document context."
        )

    def test_load_dataset_reports_line_for_invalid_question(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "eval.jsonl"
            dataset_path.write_text('{"question":"ok"}\n{"answer":"missing"}\n')

            with self.assertRaisesRegex(ValueError, "line 2"):
                load_dataset(dataset_path)

    def test_collect_outputs_include_page_aware_context(self):
        workflow = Mock()
        workflow.invoke.return_value = {
            "answer": "Entropy is discussed on this page.",
            "retrieved_docs": [
                Document(
                    page_content="Entropy details.",
                    metadata={"source": "notes.pdf", "page": 4},
                )
            ],
        }

        rows = collect_rag_outputs(
            [{"question": "What is entropy?", "reference_answer": None}],
            workflow_runner=workflow,
        )

        self.assertEqual(
            rows[0]["contexts"],
            ["[Source: notes.pdf, page 4]\nEntropy details."],
        )
        workflow.invoke.assert_called_once_with(
            {"query": "What is entropy?", "use_memory": False}
        )

    def test_evaluation_splits_optional_references_and_aggregates_scores(self):
        workflow = Mock()
        workflow.invoke.side_effect = [
            {"answer": "Answer one.", "retrieved_docs": []},
            {"answer": "Answer two.", "retrieved_docs": []},
        ]
        evaluator = Mock(
            side_effect=[
                SimpleNamespace(
                    scores=[
                        {"faithfulness": 0.8, "answer_relevancy": 0.9},
                        {"faithfulness": 0.6, "answer_relevancy": 0.7},
                    ]
                ),
                SimpleNamespace(
                    scores=[
                        {
                            "context_precision": 0.5,
                            "context_recall": 1.0,
                            "answer_correctness": 0.75,
                        }
                    ]
                ),
            ]
        )

        with (
            patch(
                "backend.ragas_evaluation.LangchainLLMWrapper",
                side_effect=lambda value: value,
            ),
            patch(
                "backend.ragas_evaluation.LangchainEmbeddingsWrapper",
                side_effect=lambda value: value,
            ),
            patch(
                "backend.ragas_evaluation.get_embedding_function",
                return_value=lambda _: [0.1, 0.2],
            ),
        ):
            report = evaluate_dataset(
                [
                    {"question": "Question one?", "reference_answer": "Reference."},
                    {"question": "Question two?", "reference_answer": None},
                ],
                workflow_runner=workflow,
                ragas_evaluator=evaluator,
                llm=Mock(),
            )

        self.assertEqual(evaluator.call_count, 2)
        base_dataset = evaluator.call_args_list[0].kwargs["dataset"]
        reference_dataset = evaluator.call_args_list[1].kwargs["dataset"]
        self.assertEqual(len(base_dataset), 2)
        self.assertEqual(
            set(base_dataset.column_names),
            {"user_input", "response", "retrieved_contexts"},
        )
        self.assertEqual(len(reference_dataset), 1)
        self.assertEqual(reference_dataset[0]["reference"], "Reference.")
        self.assertEqual(
            report["summary"],
            {
                "faithfulness": 0.7,
                "answer_relevancy": 0.8,
                "context_precision": 0.5,
                "context_recall": 1.0,
                "answer_correctness": 0.75,
            },
        )
        self.assertNotIn(
            "answer_correctness", report["examples"][1]["scores"]
        )

    def test_invalid_ragas_scores_raise_instead_of_reporting_success(self):
        workflow = Mock()
        workflow.invoke.return_value = {"answer": "Answer.", "retrieved_docs": []}
        evaluator = Mock(
            return_value=SimpleNamespace(
                scores=[
                    {"faithfulness": None, "answer_relevancy": 0.7}
                ]
            )
        )

        with (
            patch(
                "backend.ragas_evaluation.LangchainLLMWrapper",
                side_effect=lambda value: value,
            ),
            patch(
                "backend.ragas_evaluation.LangchainEmbeddingsWrapper",
                side_effect=lambda value: value,
            ),
            patch(
                "backend.ragas_evaluation.get_embedding_function",
                return_value=lambda _: [0.1, 0.2],
            ),
            self.assertRaisesRegex(ValueError, "numeric faithfulness score"),
        ):
            evaluate_dataset(
                [{"question": "Question?", "reference_answer": None}],
                workflow_runner=workflow,
                ragas_evaluator=evaluator,
                llm=Mock(),
            )


if __name__ == "__main__":
    unittest.main()
