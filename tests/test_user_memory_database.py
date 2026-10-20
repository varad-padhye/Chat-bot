import unittest
from unittest.mock import MagicMock, patch

from backend.database import (
    clear_user_memories,
    delete_user_memory,
    list_user_memories,
    search_user_memories,
    similarity_search,
    upsert_user_memory,
)


class UserMemoryDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock()
        self.cursor = MagicMock()
        self.connection.__enter__.return_value = self.connection
        self.connection.cursor.return_value.__enter__.return_value = self.cursor
        self.connection.cursor.return_value.__exit__.return_value = None
        self.connection_factory = patch(
            "backend.database.get_connection", return_value=self.connection
        )
        self.connection_factory.start()
        self.addCleanup(self.connection_factory.stop)
        self.embedding_function = patch(
            "backend.embeddings.get_embedding_function",
            return_value=lambda _: [0.1, 0.2],
        )
        self.embedding_function.start()
        self.addCleanup(self.embedding_function.stop)

    def test_memory_upsert_is_keyed_by_user_and_replaces_old_values(self):
        self.cursor.fetchone.return_value = (42,)

        memory_id = upsert_user_memory(
            "student-1", "Explanation_Style", "I prefer concise explanations."
        )

        self.assertEqual(memory_id, 42)
        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("ON CONFLICT (user_id, memory_key)", sql)
        self.assertEqual(
            parameters,
            (
                "student-1",
                "explanation_style",
                "I prefer concise explanations.",
                "[0.1,0.2]",
            ),
        )

    def test_new_memories_stop_at_per_user_limit(self):
        self.cursor.fetchone.side_effect = [None, (100,)]

        with self.assertRaisesRegex(ValueError, "limit reached"):
            upsert_user_memory("student-1", "new_key", "A new preference.")

        self.assertEqual(self.cursor.execute.call_count, 2)
        self.assertFalse(
            any(
                "INSERT INTO user_memories" in call.args[0]
                for call in self.cursor.execute.call_args_list
            )
        )

    def test_semantic_search_filters_by_user(self):
        self.cursor.fetchall.return_value = [
            ("I prefer concise explanations.", "explanation_style")
        ]

        memories = search_user_memories("student-1", "answer style")

        self.assertEqual(memories[0].page_content, "I prefer concise explanations.")
        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("WHERE user_id = %s", sql)
        self.assertEqual(parameters[0], "student-1")

    def test_document_retrieval_preserves_page_and_source_metadata(self):
        self.cursor.fetchall.return_value = [
            ("lecture.pdf", 4, "Retrieved page content.", {"page": 12})
        ]

        with patch(
            "backend.embeddings.get_embedding_function",
            return_value=lambda _: [0.1, 0.2],
        ):
            documents = similarity_search("page content")

        self.assertEqual(documents[0].metadata["source"], "lecture.pdf")
        self.assertEqual(documents[0].metadata["page"], 12)
        self.assertEqual(documents[0].metadata["chunk_index"], 4)
        self.assertEqual(documents[0].page_content, "Retrieved page content.")

    def test_memory_management_is_user_scoped(self):
        self.cursor.fetchall.return_value = [
            (3, "style", "I prefer concise answers.", "created", "updated")
        ]
        self.assertEqual(list_user_memories("student-1")[0]["id"], 3)
        self.assertEqual(self.cursor.execute.call_args.args[1], ("student-1",))

        self.cursor.rowcount = 1
        self.assertTrue(delete_user_memory("student-1", 3))
        sql, parameters = self.cursor.execute.call_args.args
        self.assertEqual(parameters, (3, "student-1"))
        self.assertIn("user_id = %s", sql)

    def test_clear_memories_does_not_delete_chats_or_documents(self):
        self.cursor.rowcount = 2

        count = clear_user_memories("student-1")

        self.assertEqual(count, 4)
        statements = [
            call.args[0]
            for call in self.cursor.execute.call_args_list
        ]
        self.assertTrue(any("DELETE FROM user_memories" in sql for sql in statements))
        self.assertTrue(
            any("DELETE FROM conversation_memories" in sql for sql in statements)
        )
        self.assertFalse(
            any(
                any(table in sql for table in ("conversations", "documents"))
                for sql in statements
            )
        )


if __name__ == "__main__":
    unittest.main()
