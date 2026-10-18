import unittest
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.workflow import (
    generate_answer,
    resolve_follow_up,
    retrieve_docs,
    workflow,
)


class FollowUpRetrievalTests(unittest.TestCase):
    def test_follow_up_is_rewritten_using_the_current_thread_history(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(
            content="What are the disadvantages of transformers?"
        )
        state = {
            "query": "What are their disadvantages?",
            "chat_history": [
                {"role": "user", "content": "Explain transformers."},
                {"role": "assistant", "content": "Transformers use attention."},
            ],
        }

        with patch("backend.workflow.get_llm", return_value=llm):
            result = resolve_follow_up(state)

        self.assertEqual(
            result["retrieval_query"],
            "What are the disadvantages of transformers?",
        )
        llm.invoke.assert_called_once()

    def test_first_turn_uses_original_query_without_calling_llm(self):
        with patch("backend.workflow.get_llm") as get_llm:
            result = resolve_follow_up({"query": "Explain transformers."})

        self.assertEqual(result["retrieval_query"], "Explain transformers.")
        get_llm.assert_not_called()

    def test_disabling_long_term_memory_keeps_document_retrieval(self):
        with (
            patch("backend.workflow.similarity_search", return_value=["document"]) as docs,
            patch("backend.workflow.search_user_memories") as memories,
        ):
            result = retrieve_docs(
                {
                    "query": "What is entropy?",
                    "retrieval_query": "Define entropy.",
                    "use_memory": False,
                }
            )

        docs.assert_called_once_with("Define entropy.", k=5)
        memories.assert_not_called()
        self.assertEqual(result["retrieved_docs"], ["document"])
        self.assertEqual(result["memory_docs"], [])

    def test_enabled_memory_is_scoped_to_user_and_does_not_replace_documents(self):
        with (
            patch("backend.workflow.similarity_search", return_value=["document"]),
            patch(
                "backend.workflow.search_user_memories",
                return_value=["preference"],
            ) as memories,
        ):
            result = retrieve_docs(
                {
                    "query": "Explain briefly.",
                    "retrieval_query": "Explain briefly.",
                    "user_id": "student-2",
                    "use_memory": True,
                }
            )

        memories.assert_called_once_with("student-2", "Explain briefly.", k=3)
        self.assertEqual(result["retrieved_docs"], ["document"])
        self.assertEqual(result["memory_docs"], ["preference"])

    def test_missing_provider_credentials_are_reported(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
            from backend.workflow import get_llm

            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                get_llm()

    def test_short_term_history_is_used_when_long_term_memory_is_disabled(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="Attention can be costly.")
        state = {
            "query": "What are their disadvantages?",
            "chat_history": [
                {"role": "user", "content": "Explain transformers."},
                {"role": "assistant", "content": "They use self-attention."},
            ],
            "retrieved_docs": [],
            "memory_docs": [],
        }

        with patch("backend.workflow.get_llm", return_value=llm):
            result = generate_answer(state)

        self.assertEqual(result["answer"], "Attention can be costly.")
        prompt_messages = llm.invoke.call_args.args[0]
        self.assertTrue(
            any(
                "Explain transformers." in message.content
                for message in prompt_messages
            )
        )
        self.assertTrue(
            any(
                "No relevant previous conversations." in message.content
                for message in prompt_messages
            )
        )

    def test_graph_accepts_stable_thread_id_and_completes_without_memory(self):
        llm = Mock()
        llm.invoke.return_value = SimpleNamespace(content="Answer from mock.")
        with (
            patch("backend.workflow.similarity_search", return_value=[]),
            patch("backend.workflow.get_llm", return_value=llm),
        ):
            result = workflow.invoke(
                {
                    "query": "Explain entropy.",
                    "thread_id": "stable-thread-id",
                    "use_memory": False,
                },
                config={"configurable": {"thread_id": "stable-thread-id"}},
            )

        self.assertEqual(result["answer"], "Answer from mock.")


if __name__ == "__main__":
    unittest.main()
