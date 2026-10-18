import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.memory import parse_memory_candidates, remember_user_message


class LongTermMemoryTests(unittest.TestCase):
    def test_parse_candidates_rejects_secret_content(self):
        candidates = parse_memory_candidates(
            '[{"key":"style","text":"I prefer concise answers."},'
            '{"key":"credential","text":"My API key is abc"},'
            '{"key":"token","text":"sk-123456789012345678901234"}]'
        )

        self.assertEqual(
            candidates,
            [{"key": "style", "text": "I prefer concise answers."}],
        )

    def test_transient_question_does_not_trigger_memory_extraction(self):
        llm = Mock()

        saved = remember_user_message("student-1", "What is entropy?", llm=llm)

        self.assertEqual(saved, 0)
        llm.invoke.assert_not_called()

    def test_explicit_preference_is_extracted_and_upserted(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content='[{"key":"explanation_style","text":'
            '"I prefer concise explanations."}]'
        )
        with patch(
            "backend.memory.upsert_user_memory", return_value=23
        ) as upsert:
            saved = remember_user_message(
                "student-1", "I prefer concise explanations.", llm=llm
            )

        self.assertEqual(saved, 1)
        upsert.assert_called_once_with(
            "student-1",
            "explanation_style",
            "I prefer concise explanations.",
        )

    def test_messages_containing_sensitive_terms_are_not_sent_to_model(self):
        llm = Mock()

        saved = remember_user_message(
            "student-1",
            "Remember my API key is sensitive-secret-value.",
            llm=llm,
        )

        self.assertEqual(saved, 0)
        llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
