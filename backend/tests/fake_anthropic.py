"""Test doubles for the Anthropic provider and the responder seam. Tests only.

``FakeAnthropic`` stands in for ``AsyncAnthropic``: it answers the three kinds of call the bot
makes (understand_turn, answer drafting with native citations, assess_answer) from scripted
data and records every request. ``ScriptedResponder`` is a responder-level double for flow
tests that do not care about request shapes. Neither is importable from production code.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from anthropic.types import Message as ProviderMessage

from app.llm.prompts import NO_ANSWER
from app.llm.understanding import TurnUnderstanding

UNKNOWN_REPLY = "I don't have verified information about that topic."


def understanding(**overrides: Any) -> dict:
    """A full, valid understanding payload; override only what a test cares about."""
    base: dict[str, Any] = {
        "language": "en",
        "intent": "question",
        "relation": "standalone",
        "query": "",
        "ambiguity": "",
        "matches_handoff_topic": False,
        "reply": "",
    }
    base.update(overrides)
    # The "no verified information" text is written by its own call; scripts may still name it here.
    base.pop("unknown_reply", None)
    return base


def scripted_unknown_reply(value: dict | None, default: str = UNKNOWN_REPLY) -> str:
    return str((value or {}).get("unknown_reply") or default)


def assessment(status: str = "answered", reason: str = "responsive", **extra: Any) -> dict:
    return {"status": status, "reason": reason, **extra}


@dataclass
class Draft:
    """What the fake drafting model returns: text, and the passage it quotes as its citation."""

    text: str
    proof: str | None = None
    # Further text after the cited part, with no citation (e.g. an honest limitation).
    tail: str = ""


NO_ANSWER_DRAFT = Draft(NO_ANSWER)


def _tool_response(name: str, value: dict) -> Any:
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[SimpleNamespace(type="tool_use", name=name, input=value)],
    )


def _documents(kwargs: dict) -> list[dict]:
    return [
        block
        for message in kwargs.get("messages", [])
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "document"
    ]


def _payload(kwargs: dict) -> dict:
    import json

    return json.loads(kwargs["messages"][0]["content"])


def _pick(source: Any, payload_or_kwargs: Any, stage: str) -> Any:
    if callable(source):
        return source(payload_or_kwargs)
    if isinstance(source, list):
        if not source:
            raise AssertionError(f"no scripted {stage} response left")
        return source.pop(0)
    return source


class FakeAnthropic:
    """``AsyncAnthropic``-compatible: ``client.messages.create(**kwargs)``."""

    def __init__(
        self,
        *,
        understand: dict | list | Callable[[dict], dict] | None = None,
        draft: Draft | list | Callable[[list[dict]], Draft] | None = None,
        assess: dict | list | Callable[[dict], dict] | None = None,
        fail: set[str] | None = None,
    ) -> None:
        self.understand = understand
        self.draft = draft
        self.assess = assess
        self.fail = fail or set()
        self.calls: list[dict] = []
        self.unknown_reply = UNKNOWN_REPLY
        self.messages = self

    async def close(self) -> None:
        return None

    def stage_calls(self, stage: str) -> list[dict]:
        return [call for call in self.calls if call["stage"] == stage]

    async def create(self, **kwargs: Any) -> Any:
        tool = (kwargs.get("tool_choice") or {}).get("name")
        stage = tool or "draft"
        if (
            stage == "draft"
            and len(kwargs["messages"]) > 1
            and "unverified_draft" in str(kwargs["messages"][-1]["content"])
        ):
            stage = "repair"
        self.calls.append({"stage": stage, **kwargs})
        if stage in self.fail or (stage == "repair" and "draft" in self.fail):
            raise TimeoutError(f"scripted {stage} failure")
        if tool == "understand_turn":
            payload = _payload(kwargs)
            value = _pick(self.understand, payload, stage)
            self.unknown_reply = scripted_unknown_reply(value, self.unknown_reply)
            value = understanding(**(value or {}))
            if not value["query"]:
                value["query"] = payload["latest_message"]
            return _tool_response(tool, value)
        if tool == "write_unknown_reply":
            return _tool_response(tool, {"reply": self.unknown_reply})
        if tool == "assess_answer":
            value = _pick(self.assess, _payload(kwargs), stage)
            return _tool_response(tool, value or assessment())
        return self._draft_response(kwargs, stage)

    def _draft_response(self, kwargs: dict, stage: str) -> Any:
        documents = _documents(kwargs)
        draft = _pick(self.draft, documents, stage)
        if draft is None:
            raise AssertionError("the drafting model was called but no draft was scripted")
        citations: list[dict] = []
        if draft.proof:
            for index, document in enumerate(documents):
                source = document["source"]["data"]
                if draft.proof in source:
                    start = source.index(draft.proof)
                    citations = [
                        {
                            "type": "char_location",
                            "document_index": index,
                            "document_title": document["title"],
                            "start_char_index": start,
                            "end_char_index": start + len(draft.proof),
                            "cited_text": draft.proof,
                        }
                    ]
                    break
            assert citations, "the quoted passage was not among the retrieved documents"
        return ProviderMessage.model_validate(
            {
                "id": "msg_fake",
                "type": "message",
                "role": "assistant",
                "model": "fake",
                "stop_reason": "end_turn",
                "content": [
                    {"type": "text", "text": draft.text, "citations": citations},
                    *([{"type": "text", "text": draft.tail}] if draft.tail else []),
                ],
                "usage": {"input_tokens": 100, "output_tokens": 30},
            }
        )


class ResponderDouble:
    """Understanding, assessment and repair for responder-level doubles; subclasses draft."""

    understand: Any = None
    assess: Any = None

    async def understand_turn(self, inputs: Any) -> TurnUnderstanding:
        self.calls.append({"stage": "understand_turn", "message": inputs.latest_message})
        value = self.understand(inputs) if callable(self.understand) else (self.understand or {})
        self.unknown_reply = scripted_unknown_reply(value, self.unknown_reply)
        value = understanding(**value)
        if not value["query"]:
            value["query"] = inputs.latest_message
        return TurnUnderstanding.model_validate(value)

    unknown_reply: str = UNKNOWN_REPLY

    async def write_unknown_reply(self, turn) -> str:
        self.calls.append({"stage": "write_unknown_reply"})
        return self.unknown_reply

    async def repair_grounded_draft(self, turn, documents, draft, failure, **_kwargs):
        self.calls.append({"stage": "repair", "failure": failure})
        return None

    async def assess_answer(self, turn, decision):
        from app.llm.answer_relevance import AnswerAssessment

        self.calls.append({"stage": "assess_answer"})
        value = self.assess(turn, decision) if callable(self.assess) else self.assess
        return AnswerAssessment.model_validate(value or assessment())

    def stage_calls(self, stage: str) -> list[dict]:
        return [call for call in self.calls if call["stage"] == stage]


@dataclass
class ScriptedResponder(ResponderDouble):
    """Scripted understanding, drafting and assessment; no request shapes involved."""

    understand: dict | Callable[[Any], dict] | None = None
    draft: Callable[[Any, list], Any] | None = None
    assess: dict | Callable[[Any, Any], dict] | None = None
    calls: list[dict] = field(default_factory=list)

    async def generate_grounded_draft(self, turn, documents, **_kwargs):
        self.calls.append(
            {
                "stage": "draft",
                "visitor_text": turn.visitor_text,
                "document_ids": [item.id for item in documents],
            }
        )
        if self.draft is None:
            raise AssertionError("the drafting model was called but no draft was scripted")
        return self.draft(turn, documents)
