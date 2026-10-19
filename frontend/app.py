import logging
import os
import tempfile
from pathlib import Path
import sys

import psycopg2
import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

load_dotenv(ROOT_DIR / ".env")
st.set_page_config(page_title="Personal RAG Chatbot", page_icon=":books:")
st.title(":books: Study Assistant – Retrieval‑Augmented Generation Chatbot")

try:
    deployment_secrets = dict(st.secrets)
except StreamlitSecretNotFoundError:
    deployment_secrets = {}

for secret_name in (
    "GOOGLE_API_KEY",
    "GEMINI_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "LLM_PROVIDER",
    "OLLAMA_MODEL",
    "EMBEDDING_MODEL",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_SSLMODE",
    "RAG_USER_ID",
    "CHAT_HISTORY_TURNS",
):
    if secret_name in deployment_secrets:
        os.environ.setdefault(secret_name, str(deployment_secrets[secret_name]))

from backend.database import (
    clear_user_memories,
    create_conversation,
    delete_conversation,
    delete_user_memory,
    get_conversation_messages,
    init_db,
    list_conversations,
    list_user_memories,
    rename_conversation,
    save_conversation_message,
)
from backend.ingestion import main as ingest_main
from backend.memory import remember_user_message
from backend.workflow import workflow

RAG_USER_ID = os.getenv("RAG_USER_ID", "default").strip() or "default"
SOURCE_CONTEXT_MARKER = "\n\n---\n\n**Sources and retrieved context**\n"


def _with_source_context(answer: str, documents: list) -> str:
    source_entries = []
    for document in documents:
        metadata = getattr(document, "metadata", {})
        if not isinstance(metadata, dict):
            continue
        source = metadata.get("source")
        if isinstance(source, str) and source.strip():
            source = source.strip()
            if "://" not in source:
                source = Path(source.replace("\\", "/")).name or source
        else:
            source = "Unknown source"

        chunk_index = metadata.get("chunk_index")
        chunk_label = (
            f" (chunk {chunk_index + 1})"
            if isinstance(chunk_index, int)
            else ""
        )
        excerpt = " ".join(
            str(getattr(document, "page_content", "")).split()
        )
        if len(excerpt) > 300:
            excerpt = f"{excerpt[:297]}..."
        source_entries.append(
            f"- **{source}**{chunk_label}\n  > {excerpt}"
        )

    if not source_entries:
        source_entries.append("No document chunks were retrieved.")
    return f"{answer}{SOURCE_CONTEXT_MARKER}" + "\n".join(source_entries)


def _render_assistant_message(content: str) -> None:
    answer, marker, source_context = content.partition(SOURCE_CONTEXT_MARKER)
    st.markdown(answer)
    if marker:
        with st.expander("Sources and retrieved context"):
            st.markdown(source_context)


@st.cache_resource(show_spinner=False)
def initialize_database() -> bool:
    init_db()
    return True


def _new_chat(user_id: str) -> None:
    conversation = create_conversation(user_id)
    st.session_state.active_thread_id = conversation["id"]
    st.rerun()


try:
    initialize_database()
    conversations = list_conversations(RAG_USER_ID)
except psycopg2.Error:
    logging.exception("Could not initialize or access the chatbot database.")
    st.error(
        "Could not connect to PostgreSQL. Check the database service and "
        "connection settings, then reload the app."
    )
    st.stop()

conversation_ids = {conversation["id"] for conversation in conversations}
active_thread_id = st.session_state.get("active_thread_id")
if active_thread_id not in conversation_ids:
    if conversations:
        st.session_state.active_thread_id = conversations[0]["id"]
    else:
        try:
            conversation = create_conversation(RAG_USER_ID)
            st.session_state.active_thread_id = conversation["id"]
        except psycopg2.Error:
            logging.exception("Could not create the initial conversation.")
            st.error("Could not create a conversation. Check PostgreSQL and reload.")
            st.stop()
    st.rerun()

active_thread_id = st.session_state.active_thread_id
try:
    active_messages = get_conversation_messages(RAG_USER_ID, active_thread_id)
except psycopg2.Error:
    logging.exception("Could not load the active conversation.")
    st.error("Could not load this conversation from PostgreSQL.")
    st.stop()

try:
    history_turns = int(os.getenv("CHAT_HISTORY_TURNS", "8"))
except ValueError:
    history_turns = 8
history_turns = min(max(history_turns, 1), 50)

if st.session_state.pop("_reset_clear_memory_confirmation", False):
    st.session_state["confirm_clear_memories"] = False

with st.sidebar:
    st.header("Chats")
    if st.button(
        "+ New Chat",
        key="new_chat",
        type="primary",
        width="stretch",
    ):
        try:
            _new_chat(RAG_USER_ID)
        except psycopg2.Error:
            logging.exception("Could not create a new conversation.")
            st.error("Could not create a new chat. Check PostgreSQL and try again.")

    for conversation in conversations:
        conversation_id = conversation["id"]
        selected = conversation_id == active_thread_id
        title_column, menu_column = st.columns([0.84, 0.16])
        if title_column.button(
            conversation["title"],
            key=f"open_{conversation_id}",
            type="primary" if selected else "secondary",
            width="stretch",
        ):
            st.session_state.active_thread_id = conversation_id
            st.rerun()

        with menu_column.popover("⋯", help="Rename or delete this conversation"):
            rename_key = f"rename_input_{conversation_id}"
            reset_rename_key = f"_reset_rename_{conversation_id}"
            if st.session_state.pop(reset_rename_key, False):
                st.session_state.pop(rename_key, None)
            new_title = st.text_input(
                "Conversation title",
                value=conversation["title"],
                key=rename_key,
                max_chars=80,
            )
            if st.button("Rename", key=f"rename_{conversation_id}"):
                try:
                    if rename_conversation(RAG_USER_ID, conversation_id, new_title):
                        st.session_state[reset_rename_key] = True
                        st.rerun()
                    st.error("Conversation not found.")
                except (psycopg2.Error, ValueError):
                    logging.exception("Could not rename conversation.")
                    st.error("Could not rename this conversation.")

            confirm_key = f"confirm_delete_{conversation_id}"
            if not st.session_state.get(confirm_key, False):
                if st.button("Delete…", key=f"delete_prompt_{conversation_id}"):
                    st.session_state[confirm_key] = True
                    st.rerun()
            else:
                st.warning("Delete this conversation and its messages?")
                if st.button(
                    "Confirm delete",
                    key=f"delete_confirm_{conversation_id}",
                    type="primary",
                ):
                    try:
                        if delete_conversation(RAG_USER_ID, conversation_id):
                            st.session_state.pop(confirm_key, None)
                            if conversation_id == active_thread_id:
                                st.session_state.pop("active_thread_id", None)
                            st.rerun()
                        st.error("Conversation not found.")
                    except psycopg2.Error:
                        logging.exception("Could not delete conversation.")
                        st.error("Could not delete this conversation.")
                if st.button("Cancel", key=f"delete_cancel_{conversation_id}"):
                    st.session_state.pop(confirm_key, None)
                    st.rerun()

    st.divider()
    use_memory = st.checkbox(
        "Use long-term memory",
        value=True,
        help=(
            "Retrieve saved user preferences and facts, and save explicit "
            "durable details from your messages."
        ),
    )

    with st.expander("Manage memories"):
        try:
            memories = list_user_memories(RAG_USER_ID)
        except psycopg2.Error:
            logging.exception("Could not list user memories.")
            st.error("Could not load saved memories.")
            memories = []

        if not memories:
            st.caption("No saved memories.")
        for memory in memories:
            st.write(memory["memory_text"])
            if st.button(
                "Delete memory",
                key=f"delete_memory_{memory['id']}",
            ):
                try:
                    delete_user_memory(RAG_USER_ID, memory["id"])
                    st.rerun()
                except psycopg2.Error:
                    logging.exception("Could not delete a user memory.")
                    st.error("Could not delete that memory.")

        if memories:
            confirm_clear_memories = st.checkbox(
                "Confirm clearing all memories",
                key="confirm_clear_memories",
            )
            if st.button("Clear long-term memory"):
                if not confirm_clear_memories:
                    st.warning("Confirm the checkbox before clearing memories.")
                else:
                    try:
                        deleted_count = clear_user_memories(RAG_USER_ID)
                        st.session_state[
                            "_reset_clear_memory_confirmation"
                        ] = True
                        st.success(f"Cleared {deleted_count} saved memories.")
                        st.rerun()
                    except psycopg2.Error:
                        logging.exception("Could not clear user memories.")
                        st.error("Could not clear saved memories.")

    with st.expander("Add knowledge", expanded=False):
        pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
        url_input = st.text_input("Web page URL")
        yt_url = st.text_input("YouTube video URL (captions required)")
        if st.button("Ingest"):
            if not pdf_file and not url_input and not yt_url:
                st.error("Upload a PDF or enter a web page or YouTube URL first.")
            else:
                try:
                    with st.spinner(
                        "Ingesting your document/source. This may take a moment; "
                        "please wait."
                    ):
                        if pdf_file:
                            safe_filename = Path(
                                pdf_file.name.replace("\\", "/")
                            ).name
                            with tempfile.TemporaryDirectory(
                                prefix="rag-upload-"
                            ) as tmp_dir:
                                pdf_path = Path(tmp_dir) / safe_filename
                                pdf_path.write_bytes(pdf_file.getvalue())
                                ingest_main(["--pdf", str(pdf_path)])
                        if url_input:
                            ingest_main(["--url", url_input])
                        if yt_url:
                            ingest_main(["--youtube", yt_url])
                    st.success("Document/source ingested successfully.")
                except (ValueError, OSError, psycopg2.Error):
                    logging.exception("Knowledge ingestion failed.")
                    st.error(
                        "Ingestion failed. Check that the source is valid, "
                        "captions are available, and PostgreSQL is reachable."
                    )

active_conversation = next(
    conversation
    for conversation in conversations
    if conversation["id"] == active_thread_id
)
st.subheader(active_conversation["title"])

for message in active_messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            _render_assistant_message(message["content"])
        else:
            st.write(message["content"])

if user_query := st.chat_input("Ask a question about your notes …"):
    if not user_query.strip():
        st.warning("Enter a message before sending.")
        st.stop()

    history = [
        {
            "role": message["role"],
            "content": (
                message["content"].partition(SOURCE_CONTEXT_MARKER)[0]
                if message["role"] == "assistant"
                else message["content"]
            ),
        }
        for message in active_messages[-history_turns * 2 :]
    ]
    try:
        save_conversation_message(
            RAG_USER_ID, active_thread_id, "user", user_query
        )
    except (psycopg2.Error, ValueError):
        logging.exception("Could not save the user message.")
        st.error("Your message could not be saved. Please check PostgreSQL.")
        st.stop()

    with st.chat_message("user"):
        st.write(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... searching your notes and preparing an answer."):
            try:
                result = workflow.invoke(
                    {
                        "query": user_query,
                        "chat_history": history,
                        "thread_id": active_thread_id,
                        "user_id": RAG_USER_ID,
                        "use_memory": use_memory,
                    },
                    config={"configurable": {"thread_id": active_thread_id}},
                )
                answer = result.get("answer")
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("The workflow did not return an answer.")
            except (psycopg2.Error, ValueError):
                logging.exception("The RAG workflow could not answer the request.")
                st.error(
                    "The answer could not be generated. Check the selected LLM "
                    "provider, its API key, and database availability."
                )
                st.info("Your question is saved in this conversation; you can retry it.")
                st.stop()
            except Exception:
                logging.exception("The configured LLM or RAG workflow failed.")
                st.error(
                    "The answer could not be generated. Check your LLM provider "
                    "configuration and try again."
                )
                st.info("Your question is saved in this conversation; you can retry it.")
                st.stop()

        answer = _with_source_context(
            answer, result.get("retrieved_docs", [])
        )
        _render_assistant_message(answer)

    try:
        save_conversation_message(
            RAG_USER_ID, active_thread_id, "assistant", answer
        )
    except (psycopg2.Error, ValueError):
        logging.exception("Could not save the assistant response.")
        st.error(
            "The answer was generated but could not be saved. "
            "Check PostgreSQL before continuing."
        )
        st.stop()

    if use_memory:
        try:
            saved_count = remember_user_message(RAG_USER_ID, user_query)
            if saved_count:
                st.toast(f"Saved {saved_count} long-term memory.")
        except (psycopg2.Error, ValueError):
            logging.exception("Could not extract or save long-term memory.")
            st.warning("The answer was saved, but long-term memory was not updated.")
        except Exception:
            logging.exception("Memory extraction failed.")
            st.warning("The answer was saved, but long-term memory was not updated.")

    st.rerun()
