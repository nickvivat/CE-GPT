"""
Chat history compression module.
Splits messages into 'old' and 'recent', summarizes older messages via LLM,
and returns a compressed view (summary + recent full messages) for context.
"""

from dataclasses import dataclass
from typing import Optional, List, Any, Protocol

from ..utils.config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class MessageLike(Protocol):
    """Protocol for message-like objects (e.g. ChatMessage) with role and content."""
    role: str
    content: str


@dataclass
class CompressedHistory:
    """Result of compressing chat history: optional summary of older messages + recent messages in full."""
    summary: Optional[str]
    recent_messages: List[Any]


def _format_messages_for_summary(messages: List[MessageLike]) -> str:
    """Format a list of messages as text for the summarization prompt."""
    parts = []
    for msg in messages:
        role_label = "User" if msg.role == "user" else "Assistant"
        parts.append(f"{role_label}: {msg.content}")
    return "\n".join(parts)


def _truncate_summary(summary: str, max_tokens: int) -> str:
    """Truncate summary to approximately max_tokens (using ~4 chars per token heuristic)."""
    if not summary or max_tokens <= 0:
        return summary or ""
    max_chars = max_tokens * 4
    if len(summary) <= max_chars:
        return summary
    truncated = summary[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space > 0:
        truncated = truncated[:last_space]
    return truncated + "..."


def compress(
    messages: List[MessageLike],
    recent_count: int,
    summary_max_tokens: int,
    llm_client: Any,
) -> CompressedHistory:
    """
    Split messages into old vs recent, summarize old messages via LLM, return summary + recent.

    Args:
        messages: List of messages in chronological order (oldest first).
        recent_count: Number of most recent messages to keep in full.
        summary_max_tokens: Approximate max length of the summary (used to cap output).
        llm_client: LLM client with generate(prompt) -> str (e.g. LLMClient from llm_client).

    Returns:
        CompressedHistory with summary (or None if no old messages) and recent_messages.
    """
    if not messages:
        return CompressedHistory(summary=None, recent_messages=[])

    # Split: recent = last recent_count, old = the rest
    if len(messages) <= recent_count:
        return CompressedHistory(summary=None, recent_messages=list(messages))

    old_messages = messages[:-recent_count]
    recent_messages = messages[-recent_count:]

    # Summarize old messages
    conversation_text = _format_messages_for_summary(old_messages)
    prompt = (
        "Summarize the following conversation in 2-4 concise sentences. "
        "Preserve course codes (8-digit codes like 01076140, 01076311, 01076312) and key facts the user asked about. "
        "Write in the same language as the conversation (English or Thai).\n\n"
        "Conversation:\n"
        f"{conversation_text}"
    )

    summary = None
    try:
        generate_fn = getattr(llm_client, "generate", None)
        is_available_fn = getattr(llm_client, "is_available", None)
        if llm_client and callable(generate_fn) and callable(is_available_fn) and is_available_fn():
            raw_summary = generate_fn(prompt, temperature=0.3)
            if raw_summary and raw_summary.strip():
                summary = _truncate_summary(raw_summary.strip(), summary_max_tokens)
                logger.debug("Compressed %d old messages into summary (%d chars)", len(old_messages), len(summary or ""))
        else:
            logger.warning("LLM not available for history compression; returning recent messages only")
    except Exception as e:
        logger.warning("History summarization failed: %s; returning recent messages only", e)

    return CompressedHistory(summary=summary, recent_messages=recent_messages)
