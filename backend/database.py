import os
from uuid import uuid4
from typing import List, Optional, Sequence

import psycopg2
from psycopg2.extras import Json, execute_values
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# PostgreSQL connection utilities
# ---------------------------------------------------------------------------

def get_connection():
    """Create a new psycopg2 connection using env vars.

    Required environment variables:
        POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST (optional, default 'localhost'),
        POSTGRES_PORT (optional, default '5432'), POSTGRES_SSLMODE (optional, default 'prefer')
    """
    user = os.getenv("POSTGRES_USER", "rag_user")
    password = os.getenv("POSTGRES_PASSWORD", "rag_password")
    db = os.getenv("POSTGRES_DB", "rag_db")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    sslmode = os.getenv("POSTGRES_SSLMODE", "prefer")
    return psycopg2.connect(
        dbname=db,
        user=user,
        password=password,
        host=host,
        port=port,
        sslmode=sslmode,
    )


def init_db():
    """Create the required tables for the RAG system.

    The schema consists of document, conversation, and memory tables:
    * ``documents`` – stores raw text chunks and metadata.
    * ``embeddings`` – stores the pgvector embedding alongside a foreign key to ``documents``.
    * ``conversation_memories`` – stores searchable prior conversations per user.
    * ``conversations`` and ``conversation_messages`` – persist isolated chat threads.
    * ``user_memories`` – stores curated, searchable user preferences and facts.
    """
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        source TEXT NOT NULL,
                        chunk_index INT NOT NULL,
                        text TEXT NOT NULL,
                        metadata JSONB
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS embeddings (
                        document_id INT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
                        embedding VECTOR
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_memories (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        embedding VECTOR,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS conversation_memories_user_id_idx
                    ON conversation_memories (user_id);
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversations (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        title TEXT NOT NULL DEFAULT 'New chat',
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
                    ON conversations (user_id, updated_at DESC);
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_messages (
                        id BIGSERIAL PRIMARY KEY,
                        conversation_id TEXT NOT NULL
                            REFERENCES conversations(id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                        content TEXT NOT NULL CHECK (length(trim(content)) > 0),
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS conversation_messages_thread_idx
                    ON conversation_messages (conversation_id, id);
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_memories (
                        id BIGSERIAL PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        memory_key TEXT NOT NULL,
                        memory_text TEXT NOT NULL,
                        embedding VECTOR,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE (user_id, memory_key)
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS user_memories_user_idx
                    ON user_memories (user_id);
                    """
                )
                cur.execute(
                    """
                    SELECT format_type(atttypid, atttypmod)
                    FROM pg_attribute
                    WHERE attrelid = 'embeddings'::regclass
                      AND attname = 'embedding'
                      AND NOT attisdropped;
                    """
                )
                embedding_type = cur.fetchone()[0]
                if embedding_type != "vector":
                    cur.execute(
                        """
                        ALTER TABLE embeddings
                        ALTER COLUMN embedding TYPE VECTOR
                        USING embedding::VECTOR;
                        """
                    )
    finally:
        conn.close()
    print(" Database schema initialized")


def _validate_user_id(user_id: str) -> None:
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must not be empty.")


def create_conversation(user_id: str) -> dict:
    """Create a persistent, user-scoped conversation with a stable thread ID."""
    _validate_user_id(user_id)
    conversation_id = str(uuid4())
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (id, user_id)
                    VALUES (%s, %s)
                    RETURNING id, user_id, title, created_at, updated_at;
                    """,
                    (conversation_id, user_id),
                )
                row = cur.fetchone()
        return dict(zip(("id", "user_id", "title", "created_at", "updated_at"), row))
    finally:
        conn.close()


def list_conversations(user_id: str) -> List[dict]:
    """List a user's conversations, most recently updated first."""
    _validate_user_id(user_id)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = %s
                ORDER BY updated_at DESC, created_at DESC;
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        return [
            dict(zip(("id", "user_id", "title", "created_at", "updated_at"), row))
            for row in rows
        ]
    finally:
        conn.close()


def conversation_title(first_message: str, max_length: int = 48) -> str:
    """Create a compact display title from the first user message."""
    normalized = " ".join(first_message.split())
    if not normalized:
        return "New chat"
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def save_conversation_message(
    user_id: str, conversation_id: str, role: str, content: str
) -> int:
    """Append a message only to a conversation owned by the given user."""
    _validate_user_id(user_id)
    if role not in {"user", "assistant"}:
        raise ValueError("role must be 'user' or 'assistant'.")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Message content must not be empty.")

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_messages (conversation_id, role, content)
                    SELECT id, %s, %s
                    FROM conversations
                    WHERE id = %s AND user_id = %s
                    RETURNING id;
                    """,
                    (role, content.strip(), conversation_id, user_id),
                )
                row = cur.fetchone()
                if row is None:
                    raise ValueError("Conversation not found for this user.")
                message_id = row[0]
                if role == "user":
                    cur.execute(
                        """
                        UPDATE conversations
                        SET title = CASE
                                WHEN title = 'New chat' THEN %s
                                ELSE title
                            END,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s AND user_id = %s;
                        """,
                        (conversation_title(content), conversation_id, user_id),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE conversations
                        SET updated_at = CURRENT_TIMESTAMP
                        WHERE id = %s AND user_id = %s;
                        """,
                        (conversation_id, user_id),
                    )
        return message_id
    finally:
        conn.close()


def get_conversation_messages(user_id: str, conversation_id: str) -> List[dict]:
    """Load messages only when the requested conversation belongs to the user."""
    _validate_user_id(user_id)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT m.role, m.content, m.created_at
                FROM conversation_messages AS m
                JOIN conversations AS c ON c.id = m.conversation_id
                WHERE c.id = %s AND c.user_id = %s
                ORDER BY m.id;
                """,
                (conversation_id, user_id),
            )
            rows = cur.fetchall()
        return [
            {"role": role, "content": content, "created_at": created_at}
            for role, content, created_at in rows
        ]
    finally:
        conn.close()


def rename_conversation(user_id: str, conversation_id: str, title: str) -> bool:
    """Rename one conversation belonging to the user."""
    _validate_user_id(user_id)
    clean_title = " ".join(title.split())
    if not clean_title:
        raise ValueError("Conversation title must not be empty.")
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE conversations
                    SET title = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s AND user_id = %s;
                    """,
                    (clean_title[:80], conversation_id, user_id),
                )
                return cur.rowcount == 1
    finally:
        conn.close()


def delete_conversation(user_id: str, conversation_id: str) -> bool:
    """Delete one owned conversation; its messages are removed by cascade."""
    _validate_user_id(user_id)
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversations WHERE id = %s AND user_id = %s;",
                    (conversation_id, user_id),
                )
                return cur.rowcount == 1
    finally:
        conn.close()


def list_user_memories(user_id: str) -> List[dict]:
    """Return saved long-term memories for one user."""
    _validate_user_id(user_id)
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, memory_key, memory_text, created_at, updated_at
                FROM user_memories
                WHERE user_id = %s
                ORDER BY updated_at DESC;
                """,
                (user_id,),
            )
            rows = cur.fetchall()
        return [
            {
                "id": memory_id,
                "memory_key": memory_key,
                "memory_text": memory_text,
                "created_at": created_at,
                "updated_at": updated_at,
            }
            for memory_id, memory_key, memory_text, created_at, updated_at in rows
        ]
    finally:
        conn.close()


def upsert_user_memory(user_id: str, memory_key: str, memory_text: str) -> int:
    """Create or update a keyed memory, replacing stale values for that key."""
    _validate_user_id(user_id)
    key = memory_key.strip().lower()
    text = memory_text.strip()
    if not key or not text:
        raise ValueError("Memory key and text must not be empty.")
    if len(key) > 80 or len(text) > 500:
        raise ValueError("Memory key or text exceeds the allowed length.")

    from .embeddings import get_embedding_function

    embedding = get_embedding_function()(text)
    vector_literal = "[" + ",".join(str(float(value)) for value in embedding) + "]"
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM user_memories
                    WHERE user_id = %s AND memory_key = %s;
                    """,
                    (user_id, key),
                )
                existing = cur.fetchone()
                if existing is None:
                    cur.execute(
                        "SELECT COUNT(*) FROM user_memories WHERE user_id = %s;",
                        (user_id,),
                    )
                    if cur.fetchone()[0] >= 100:
                        raise ValueError(
                            "Long-term memory limit reached; delete old memories first."
                        )
                cur.execute(
                    """
                    INSERT INTO user_memories
                        (user_id, memory_key, memory_text, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (user_id, memory_key)
                    DO UPDATE SET
                        memory_text = EXCLUDED.memory_text,
                        embedding = EXCLUDED.embedding,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id;
                    """,
                    (user_id, key, text, vector_literal),
                )
                return cur.fetchone()[0]
    finally:
        conn.close()


def search_user_memories(user_id: str, query: str, k: int = 3) -> List[Document]:
    """Semantically retrieve memories scoped to one user."""
    _validate_user_id(user_id)
    if k <= 0:
        raise ValueError("k must be greater than zero.")
    from .embeddings import get_embedding_function

    query_embedding = get_embedding_function()(query)
    vector_literal = "[" + ",".join(str(float(value)) for value in query_embedding) + "]"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT memory_text, memory_key
                FROM user_memories
                WHERE user_id = %s
                  AND embedding IS NOT NULL
                  AND vector_dims(embedding) = %s
                ORDER BY embedding <=> %s::vector, updated_at DESC
                LIMIT %s;
                """,
                (user_id, len(query_embedding), vector_literal, k),
            )
            rows = cur.fetchall()
        return [
            Document(
                page_content=memory_text,
                metadata={"memory_key": memory_key},
            )
            for memory_text, memory_key in rows
        ]
    finally:
        conn.close()


def delete_user_memory(user_id: str, memory_id: int) -> bool:
    """Delete a single memory belonging to the user."""
    _validate_user_id(user_id)
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM user_memories WHERE id = %s AND user_id = %s;",
                    (memory_id, user_id),
                )
                return cur.rowcount == 1
    finally:
        conn.close()


def clear_user_memories(user_id: str) -> int:
    """Clear curated memories and legacy turn memories, leaving chats/documents intact."""
    _validate_user_id(user_id)
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM user_memories WHERE user_id = %s;", (user_id,))
                deleted = cur.rowcount
                cur.execute(
                    "DELETE FROM conversation_memories WHERE user_id = %s;",
                    (user_id,),
                )
                return deleted + cur.rowcount
    finally:
        conn.close()


def upsert_chunks(
    chunks: Sequence[Document],
    source: str,
    embeddings: Optional[Sequence[Sequence[float]]] = None,
) -> List[int]:
    """Insert a list of ``Document`` objects (from LangChain) into the DB.

    Parameters
    ----------
    chunks: List[langchain.schema.Document]
        Documents that have ``page_content`` and ``metadata``.
    source: str
        Human readable name of the origin (e.g. file name or URL).
    """
    if not chunks:
        return []
    if embeddings is not None and len(embeddings) != len(chunks):
        raise ValueError("The number of embeddings must match the number of document chunks.")

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                doc_records = [
                    (source, idx, doc.page_content, Json(doc.metadata))
                    for idx, doc in enumerate(chunks)
                ]
                rows = execute_values(
                    cur,
                    """
                    INSERT INTO documents (source, chunk_index, text, metadata)
                    VALUES %s RETURNING id;
                    """,
                    doc_records,
                    fetch=True,
                )
                ids = [row[0] for row in rows]

                if embeddings is not None:
                    execute_values(
                        cur,
                        """
                        INSERT INTO embeddings (document_id, embedding)
                        VALUES %s
                        ON CONFLICT (document_id)
                        DO UPDATE SET embedding = EXCLUDED.embedding;
                        """,
                        list(zip(ids, embeddings)),
                    )
        return ids
    finally:
        conn.close()


def similarity_search(query: str, k: int = 5) -> List[Document]:
    """Return the top matching stored chunks for a query."""
    from .embeddings import get_embedding_function

    query_embedding = get_embedding_function()(query)
    vector_literal = "[" + ",".join(str(float(value)) for value in query_embedding) + "]"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT d.text, d.metadata
                FROM documents AS d
                JOIN embeddings AS e ON e.document_id = d.id
                WHERE e.embedding IS NOT NULL
                  AND vector_dims(e.embedding) = %s
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s;
                """,
                (len(query_embedding), vector_literal, k),
            )
            rows = cur.fetchall()
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in rows
        ]
    finally:
        conn.close()


def store_conversation_memory(user_id: str, question: str, answer: str) -> int:
    """Persist a conversation turn and its vector for later semantic retrieval."""
    question = question.strip()
    answer = answer.strip()
    if not user_id.strip():
        raise ValueError("user_id must not be empty.")
    if not question or not answer:
        raise ValueError("Both question and answer must not be empty.")

    from .embeddings import get_embedding_function

    memory_text = f"Question: {question}\nAnswer: {answer}"
    embedding = get_embedding_function()(memory_text)
    vector_literal = "[" + ",".join(str(float(value)) for value in embedding) + "]"

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_memories (user_id, question, answer, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    RETURNING id;
                    """,
                    (user_id, question, answer, vector_literal),
                )
                return cur.fetchone()[0]
    finally:
        conn.close()


def search_conversation_memories(
    user_id: str, query: str, k: int = 3
) -> List[Document]:
    """Find the most relevant previous turns belonging to ``user_id``."""
    if not user_id.strip():
        raise ValueError("user_id must not be empty.")
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    from .embeddings import get_embedding_function

    query_embedding = get_embedding_function()(query)
    vector_literal = "[" + ",".join(str(float(value)) for value in query_embedding) + "]"
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT question, answer, created_at
                FROM conversation_memories
                WHERE user_id = %s
                  AND embedding IS NOT NULL
                  AND vector_dims(embedding) = %s
                ORDER BY embedding <=> %s::vector, created_at DESC
                LIMIT %s;
                """,
                (user_id, len(query_embedding), vector_literal, k),
            )
            rows = cur.fetchall()
        return [
            Document(
                page_content=f"Earlier question: {question}\nEarlier answer: {answer}",
                metadata={"created_at": created_at.isoformat()},
            )
            for question, answer, created_at in rows
        ]
    finally:
        conn.close()


def clear_conversation_memories(user_id: str) -> int:
    """Delete all stored conversation turns for one user and return the count."""
    if not user_id.strip():
        raise ValueError("user_id must not be empty.")

    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM conversation_memories WHERE user_id = %s;",
                    (user_id,),
                )
                return cur.rowcount
    finally:
        conn.close()


if __name__ == "__main__":
    # Simple CLI helper for quick DB init
    init_db()
