"""Extract explicit, durable user preferences and facts for long-term memory."""

import json
import re
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate

from .database import upsert_user_memory
from .workflow import get_llm

_EXPLICIT_MEMORY = re.compile(
    r"\b("
    r"remember(?:\s+that)?|"
    r"i\s+(?:prefer|like|love|am studying|am learning|am preparing|"
    r"am a|work as|want you to remember)|"
    r"i'm\s+(?:studying|learning|preparing|a)|"
    r"my\s+(?:goal|preference|preferred(?:\s+\w+){0,3}|major|field of study)\s+"
    r"(?:is|are|=)"
    r")\b",
    re.IGNORECASE,
)
_SENSITIVE = re.compile(
    r"\b(?:password|passcode|api[_ -]?key|access[_ -]?token|"
    r"secret|private[_ -]?key|credit[_ -]?card|cvv|ssn|"
    r"social security number|home address|date of birth|phone number)\b|"
    r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|"
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|AIza[A-Za-z0-9_-]{30,}|"
    r"gh[pousr]_[A-Za-z0-9_]{20,})\b|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"\b\d{13,19}\b",
    re.IGNORECASE,
)
_MEMORY_KEY = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$")

_EXTRACTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract only durable facts or preferences explicitly stated by the "
            "user that would improve future conversations, such as explanation "
            "style, current studies, or a goal. Save only clear statements or "
            "facts the user explicitly asks you to remember. Do not infer facts "
            "from the assistant, retain secrets or sensitive information, or "
            "save transient questions. Return a JSON array of objects with "
            "'key' (a stable short snake_case category) and 'text' (one concise "
            "first-person fact). Return [] if there is nothing appropriate.",
        ),
        ("human", "User message:\n{user_message}"),
    ]
)


def _response_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    raise ValueError("Memory extraction returned an unsupported response format.")


def parse_memory_candidates(response_text: str) -> List[Dict[str, str]]:
    """Parse and validate concise memory candidates returned by the LLM."""
    text = response_text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3].strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Memory extraction did not return valid JSON.") from exc

    if not isinstance(result, list):
        raise ValueError("Memory extraction must return a JSON array.")

    candidates_by_key: Dict[str, Dict[str, str]] = {}
    for item in result:
        if not isinstance(item, dict):
            continue
        key = item.get("key")
        memory_text = item.get("text")
        if not isinstance(key, str) or not isinstance(memory_text, str):
            continue
        key = key.strip().lower()
        memory_text = " ".join(memory_text.split())
        if (
            _MEMORY_KEY.fullmatch(key)
            and memory_text
            and len(memory_text) <= 500
            and not _SENSITIVE.search(memory_text)
        ):
            candidates_by_key[key] = {"key": key, "text": memory_text}
    return list(candidates_by_key.values())[:5]


def remember_user_message(
    user_id: str, user_message: str, llm: Optional[Any] = None
) -> int:
    """Extract and upsert suitable memories from an explicit user statement."""
    if not user_message.strip() or len(user_message) > 2000:
        return 0
    if _SENSITIVE.search(user_message) or not _EXPLICIT_MEMORY.search(user_message):
        return 0

    response = (llm or get_llm()).invoke(
        _EXTRACTION_PROMPT.format_messages(user_message=user_message)
    )
    candidates = parse_memory_candidates(_response_text(response.content))
    return sum(
        1
        for candidate in candidates
        if upsert_user_memory(user_id, candidate["key"], candidate["text"])
    )
