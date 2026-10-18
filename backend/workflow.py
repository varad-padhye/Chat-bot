import os
from typing import List, TypedDict

from langgraph.graph import END, START, StateGraph
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from .database import search_user_memories, similarity_search


class RAGState(TypedDict, total=False):
    query: str
    retrieval_query: str
    thread_id: str
    chat_history: List[dict]
    user_id: str
    use_memory: bool
    retrieved_docs: List[Document]
    memory_docs: List[Document]
    answer: str


def get_llm():
    """Select the configured Gemini, OpenAI, or Ollama chat model."""
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("Set GOOGLE_API_KEY to use the Gemini provider.")
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.0)
    if provider == "ollama":
        model_name = os.getenv("OLLAMA_MODEL", "llama3:8b")
        return ChatOllama(model=model_name)
    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY to use the OpenAI provider.")
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o")
        return ChatOpenAI(model=model_name, temperature=0.0)
    raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")


def analyze_query(state: RAGState) -> dict:
    return {"query": state["query"]}


def resolve_follow_up(state: RAGState) -> dict:
    """Rewrite follow-ups as standalone retrieval queries using bounded history."""
    history = state.get("chat_history") or []
    if not history:
        return {"retrieval_query": state["query"]}

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Rewrite the latest user message as a standalone search query "
                "using only relevant conversation context. Preserve the user's "
                "intent and language. Do not answer the question or add facts. "
                "If it is already standalone, return it unchanged. Return only "
                "the rewritten query.",
            ),
            MessagesPlaceholder(variable_name="history", optional=True),
            ("human", "Latest user message: {query}"),
        ]
    )
    messages = [
        ("human" if item.get("role") == "user" else "ai", item["content"])
        for item in history
        if item.get("role") in ("user", "assistant")
        and isinstance(item.get("content"), str)
    ]
    response = get_llm().invoke(
        prompt.format_messages(history=messages, query=state["query"])
    ).content
    if isinstance(response, list):
        response = "".join(
            block.get("text", "")
            for block in response
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    if not isinstance(response, str) or not response.strip():
        raise ValueError("The model returned an empty follow-up search query.")
    return {"retrieval_query": response.strip()}


def retrieve_docs(state: RAGState) -> dict:
    retrieval_query = state.get("retrieval_query", state["query"])
    docs = similarity_search(retrieval_query, k=5)
    memories = []
    if state.get("use_memory", False):
        memories = search_user_memories(
            state.get("user_id", "default"), retrieval_query, k=3
        )
    return {"retrieved_docs": docs, "memory_docs": memories}


def generate_answer(state: RAGState) -> dict:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an assistant helping a student study. "
                "Answer using the provided context. If the context does not "
                "contain the answer, say you do not know based on the notes. "
                "Conversation memory may help with continuity or preferences, "
                "but is not authoritative study material; rely on retrieved "
                "document context for factual answers.",
            ),
            MessagesPlaceholder(variable_name="history", optional=True),
            (
                "human",
                "Retrieved document context:\n{context}\n\n"
                "Relevant conversation memory:\n{memory_context}\n\n"
                "Question: {question}",
            ),
        ]
    )
    history = [
        ("human" if message["role"] == "user" else "ai", message["content"])
        for message in (state.get("chat_history") or [])
        if message.get("role") in ("user", "assistant")
    ]
    context = "\n---\n".join(
        document.page_content for document in (state.get("retrieved_docs") or [])
    )
    memory_context = "\n---\n".join(
        document.page_content for document in (state.get("memory_docs") or [])
    )
    messages = prompt.format_messages(
        history=history,
        context=context,
        memory_context=memory_context or "No relevant previous conversations.",
        question=state["query"],
    )
    answer = get_llm().invoke(messages).content
    return {"answer": answer}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("analyze", analyze_query)
    graph.add_node("resolve_follow_up", resolve_follow_up)
    graph.add_node("retrieve", retrieve_docs)
    graph.add_node("generate", generate_answer)
    graph.add_edge(START, "analyze")
    graph.add_edge("analyze", "resolve_follow_up")
    graph.add_edge("resolve_follow_up", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


workflow = build_graph()
