"""Conversation history as provider messages: redacted, staff turns marked."""

from __future__ import annotations

from app.models.message import Message
from app.settings import get_settings

CONVERSATION_CHAR_BUDGET = 12_000
STAFF_MARKER = "[Staff member] "


def redact_window_body(body: str) -> str:
    from app.services.pii_redactor import redact_for_model

    return redact_for_model((body or "").strip())


def conversation_window_messages(rows: list[Message], *, limit: int | None = None) -> list[dict]:
    settings = get_settings()
    size = limit if limit is not None else settings.conversation_window_size
    selected = [row for row in rows if row.role in {"visitor", "bot", "agent"}][-size:]
    messages: list[dict] = []
    for row in selected:
        role = "user" if row.role == "visitor" else "assistant"
        text = redact_window_body(row.body)
        if not text:
            continue
        if row.role == "agent":
            # Staff are not the bot: keep their words, but never as the bot's own claims.
            text = f"{STAFF_MARKER}{text}"
        messages.append({"role": role, "content": text})
    total = sum(len(message["content"]) for message in messages)
    while messages and total > CONVERSATION_CHAR_BUDGET:
        total -= len(messages.pop(0)["content"])
    return messages


def prior_provider_messages(rows: list[Message], visitor_text: str) -> list[dict]:
    messages = conversation_window_messages(rows, limit=get_settings().conversation_window_size + 1)
    current = redact_window_body(visitor_text)
    if messages and messages[-1]["role"] == "user" and messages[-1]["content"] == current:
        return messages[:-1]
    return messages
