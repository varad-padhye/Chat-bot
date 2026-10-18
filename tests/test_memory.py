import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from backend.database import (
    clear_conversation_memories,
    search_conversation_memories,
    store_conversation_memory,
)


class ConversationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock()
        self.cursor = MagicMock()
        self.connection.__enter__.return_value = self.connection
        self.connection.cursor.return_value.__enter__.return_value = self.cursor
        self.database_connection = patch(
            "backend.database.get_connection", return_value=self.connection
        )
        self.database_connection.start()
        self.addCleanup(self.database_connection.stop)

        self.embedding_function = patch(
            "backend.embeddings.get_embedding_function",
            return_value=lambda _: [0.1, 0.2],
        )
        self.embedding_function.start()
        self.addCleanup(self.embedding_function.stop)

    def test_store_conversation_memory_persists_user_and_embedding(self):
        self.cursor.fetchone.return_value = (17,)

        memory_id = store_conversation_memory(
            "student-1", "  What is entropy?  ", "A measure of disorder."
        )

        self.assertEqual(memory_id, 17)
        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("INSERT INTO conversation_memories", sql)
        self.assertEqual(parameters, (
            "student-1",
            "What is entropy?",
            "A measure of disorder.",
            "[0.1,0.2]",
        ))

    def test_search_returns_memory_scoped_to_user(self):
        created_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.cursor.fetchall.return_value = [
            ("What is entropy?", "A measure of disorder.", created_at)
        ]

        memories = search_conversation_memories("student-1", "entropy")

        self.assertEqual(len(memories), 1)
        self.assertIn("Earlier question: What is entropy?", memories[0].page_content)
        parameters = self.cursor.execute.call_args.args[1]
        self.assertEqual(parameters[0], "student-1")
        self.assertEqual(parameters[1], 2)
        self.assertEqual(parameters[-1], 3)

    def test_clear_deletes_only_the_requested_user_memories(self):
        self.cursor.rowcount = 2

        deleted_count = clear_conversation_memories("student-1")

        self.assertEqual(deleted_count, 2)
        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("DELETE FROM conversation_memories", sql)
        self.assertEqual(parameters, ("student-1",))


if __name__ == "__main__":
    unittest.main()
