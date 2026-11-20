"""Pool behaviour against a real PostgreSQL (uses POSTGRES_* env vars; skipped if unreachable)."""

import unittest

import psycopg2

from backend import database


def _backend_pid(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT pg_backend_pid();")
        return cur.fetchone()[0]


class ConnectionPoolTests(unittest.TestCase):
    def setUp(self):
        try:
            database.get_connection().close()
        except psycopg2.Error as exc:
            self.skipTest(f"PostgreSQL not reachable: {exc}")

    def test_close_returns_connection_for_reuse(self):
        conn = database.get_connection()
        pid = _backend_pid(conn)
        conn.close()
        conn = database.get_connection()
        self.assertEqual(_backend_pid(conn), pid)
        conn.close()

    def test_many_borrows_do_not_exhaust_pool(self):
        for _ in range(50):  # pool max is 10, so a leak would raise PoolError
            database.get_connection().close()

    def test_recovers_after_server_drops_idle_connection(self):
        conn = database.get_connection()
        pid = _backend_pid(conn)
        conn.close()
        killer = psycopg2.connect(
            dbname=database.os.getenv("POSTGRES_DB", "rag_db"),
            user=database.os.getenv("POSTGRES_USER", "rag_user"),
            password=database.os.getenv("POSTGRES_PASSWORD", "rag_password"),
            host=database.os.getenv("POSTGRES_HOST", "localhost"),
            port=database.os.getenv("POSTGRES_PORT", "5432"),
        )
        killer.autocommit = True
        with killer.cursor() as cur:
            cur.execute("SELECT pg_terminate_backend(%s);", (pid,))
        killer.close()

        conn = database.get_connection()
        self.assertNotEqual(_backend_pid(conn), pid)
        conn.close()

    def test_failed_transaction_does_not_poison_next_borrower(self):
        conn = database.get_connection()
        with self.assertRaises(psycopg2.Error):
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM table_that_does_not_exist;")
        conn.close()
        conn = database.get_connection()
        self.assertEqual(_backend_pid(conn) > 0, True)
        conn.close()


if __name__ == "__main__":
    unittest.main()
