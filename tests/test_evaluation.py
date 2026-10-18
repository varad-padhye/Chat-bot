import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from backend.evaluation import (
    aggregate_scores,
    evaluate_example,
    parse_judge_response,
)


class EvaluationTests(unittest.TestCase):
    def test_parse_judge_response_accepts_fenced_json(self):
        result = parse_judge_response(
            '```json\n{"scores":{"faithfulness":0.75},'
            '"reasons":{"faithfulness":"Mostly supported"}}\n```',
            ["faithfulness"],
        )

        self.assertEqual(result["scores"], {"faithfulness": 0.75})
        self.assertEqual(result["reasons"]["faithfulness"], "Mostly supported")

    def test_parse_judge_response_rejects_out_of_range_score(self):
        with self.assertRaisesRegex(ValueError, "between 0.0 and 1.0"):
            parse_judge_response(
                '{"scores":{"faithfulness":1.5}}',
                ["faithfulness"],
            )

    def test_aggregate_scores_averages_available_metrics(self):
        results = [
            {"scores": {"faithfulness": 0.5, "answer_relevance": 1.0}},
            {"scores": {"faithfulness": 1.0}},
        ]

        self.assertEqual(
            aggregate_scores(results),
            {"faithfulness": 0.75, "answer_relevance": 1.0},
        )

    def test_evaluate_example_runs_without_memory_and_scores_reference(self):
        mock_workflow = Mock()
        mock_workflow.invoke.return_value = {
            "answer": "Energy is conserved.",
            "retrieved_docs": [Document(page_content="Energy is conserved.")],
        }
        judge = Mock()
        judge.invoke.return_value = SimpleNamespace(
            content=(
                '{"scores":{"context_relevance":0.9,"faithfulness":1.0,'
                '"answer_relevance":0.8,"answer_correctness":0.9},'
                '"reasons":{}}'
            )
        )

        with patch("backend.evaluation.workflow", mock_workflow):
            result = evaluate_example(
                {
                    "question": "What is conserved?",
                    "reference_answer": "Energy is conserved.",
                },
                judge=judge,
            )

        mock_workflow.invoke.assert_called_once_with(
            {"query": "What is conserved?", "use_memory": False}
        )
        self.assertEqual(result["scores"]["answer_correctness"], 0.9)
        self.assertEqual(result["contexts"], ["Energy is conserved."])


if __name__ == "__main__":
    unittest.main()
