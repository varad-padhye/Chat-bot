import unittest
from unittest.mock import MagicMock, patch

from backend.database import (
    conversation_title,
    create_conversation,
    delete_conversation,
    get_conversation_messages,
    list_conversations,
    rename_conversation,
    save_conversation_message,
)


class ConversationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.connection = MagicMock()
        self.cursor = MagicMock()
        self.connection.__enter__.return_value = self.connection
        self.connection.cursor.return_value.__enter__.return_value = self.cursor
        self.connection.cursor.return_value.__exit__.return_value = None
        self.connection.cursor.return_value.__enter__.return_value = self.cursor
        self.connection_factory = patch(
            "backend.database.get_connection", return_value=self.connection
        )
        self.connection_factory.start()
        self.addCleanup(self.connection_factory.stop)

    def test_create_conversation_generates_stable_unique_thread_id(self):
        self.cursor.fetchone.side_effect = lambda: (
            self.cursor.execute.call_args.args[1][0],
            "student-1",
            "New chat",
            "created",
            "updated",
        )

        conversation = create_conversation("student-1")

        self.assertEqual(
            conversation["id"], self.cursor.execute.call_args.args[1][0]
        )
        self.assertEqual(len(conversation["id"]), 36)
        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("INSERT INTO conversations", sql)
        self.assertEqual(parameters[0], conversation["id"])
        self.assertEqual(parameters[1], "student-1")

    def test_messages_are_loaded_by_conversation_and_user(self):
        self.cursor.fetchall.return_value = [
            ("user", "First thread question", "time-1"),
            ("assistant", "First thread answer", "time-2"),
        ]

        messages = get_conversation_messages("student-1", "thread-1")

        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        sql, parameters = self.cursor.execute.call_args.args
        self.assertIn("c.id = %s AND c.user_id = %s", sql)
        self.assertEqual(parameters, ("thread-1", "student-1"))

    def test_saving_message_updates_title_without_crossing_threads(self):
        self.cursor.fetchone.return_value = (11,)

        message_id = save_conversation_message(
            "student-1", "thread-1", "user", "  Explain transformers to me  "
        )

        self.assertEqual(message_id, 11)
        insert_sql, insert_parameters = self.cursor.execute.call_args_list[0].args
        update_sql, update_parameters = self.cursor.execute.call_args_list[1].args
        self.assertIn("SELECT id, %s, %s", insert_sql)
        self.assertEqual(insert_parameters, (
            "user",
            "Explain transformers to me",
            "thread-1",
            "student-1",
        ))
        self.assertIn("WHERE id = %s AND user_id = %s", update_sql)
        self.assertEqual(update_parameters, (
            "Explain transformers to me",
            "thread-1",
            "student-1",
        ))

    def test_list_rename_and_delete_are_user_scoped(self):
        self.cursor.fetchall.return_value = [
            ("thread-1", "student-1", "Title", "created", "updated")
        ]
        conversations = list_conversations("student-1")
        self.assertEqual(conversations[0]["id"], "thread-1")
        self.assertEqual(
            self.cursor.execute.call_args.args[1], ("student-1",)
        )

        self.cursor.rowcount = 1
        self.assertTrue(
            rename_conversation("student-1", "thread-1", "Updated title")
        )
        self.assertTrue(delete_conversation("student-1", "thread-1"))
        delete_sql, delete_parameters = self.cursor.execute.call_args.args
        self.assertIn("DELETE FROM conversations", delete_sql)
        self.assertEqual(delete_parameters, ("thread-1", "student-1"))

    def test_empty_or_invalid_messages_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            save_conversation_message("student-1", "thread-1", "user", " ")
        with self.assertRaisesRegex(ValueError, "role must be"):
            save_conversation_message("student-1", "thread-1", "system", "text")

    def test_conversation_titles_are_compact(self):
        self.assertEqual(
            conversation_title("  Explain   transformers "),
            "Explain transformers",
        )
        self.assertLessEqual(len(conversation_title("x" * 100)), 48)


if __name__ == "__main__":
    unittest.main()
