"""TEMPORARY: telemetry helpers for the local chatbot test harness.

Nothing here alters bot behaviour. `LlmCallRecorder` wraps the real Anthropic
`messages.create` to note model/usage/latency per call; the rest turns the
request-scoped bot trace into JSON-safe, PII-redacted summaries. Remove with the harness.
"""

from __future__ import annotations

import dataclasses
import functools
import re
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from app.services.pii_redactor import redact_for_model

MAX_STR = 1500
MAX_DEPTH = 8
MAX_ITEMS = 60
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_DROP_KEYS = {"embedding", "raw_content"}

_calls: ContextVar[list[dict[str, Any]] | None] = ContextVar("harness_llm_calls", default=None)


def json_safe(value: Any, *, depth: int = 0) -> Any:  # noqa: C901
    """Convert arbitrary trace values to JSON-safe data; redact PII in free text."""
    if value is None or isinstance(value, bool | int):
        return value
    if isinstance(value, float):
        return value if value == value and abs(value) != float("inf") else None
    if isinstance(value, str):
        if _UUID_RE.match(value):
            return value
        text = redact_for_model(value)
        return text if len(text) <= MAX_STR else text[:MAX_STR] + "...[truncated]"
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    if isinstance(value, Enum):
        return json_safe(value.value, depth=depth)
    if depth >= MAX_DEPTH:
        return f"<{type(value).__name__}>"
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: json_safe(getattr(value, f.name), depth=depth + 1)
            for f in dataclasses.fields(value)
            if f.name not in _DROP_KEYS
        }
    if hasattr(value, "model_dump"):
        try:
            return json_safe(value.model_dump(), depth=depth + 1)
        except Exception:
            return f"<{type(value).__name__}>"
    if isinstance(value, dict):
        return {
            str(key): json_safe(item, depth=depth + 1)
            for key, item in list(value.items())[:200]
            if str(key) not in _DROP_KEYS
        }
    if isinstance(value, list | tuple | set | frozenset):
        items = list(value)[:MAX_ITEMS]
        out = [json_safe(item, depth=depth + 1) for item in items]
        if len(value) > MAX_ITEMS:
            out.append(f"...(+{len(value) - MAX_ITEMS} more)")
        return out
    return json_safe(str(value), depth=depth + 1)


# ---------------------------------------------------------------- LLM call recording

_installed = False


def _usage_dict(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)

    def read(name: str) -> int:
        return int(getattr(usage, name, 0) or 0)

    return {
        "input_tokens": read("input_tokens"),
        "output_tokens": read("output_tokens"),
        "cache_creation_input_tokens": read("cache_creation_input_tokens"),
        "cache_read_input_tokens": read("cache_read_input_tokens"),
    }


def make_recording_create(original: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an async `messages.create`; delegate unchanged, note the call when capturing."""

    @functools.wraps(original)
    async def create(self: Any, *args: Any, **kwargs: Any) -> Any:
        calls = _calls.get()
        if calls is None:
            return await original(self, *args, **kwargs)
        choice = kwargs.get("tool_choice")
        stage = choice.get("name") if isinstance(choice, dict) else "grounded_draft"
        record: dict[str, Any] = {
            "stage": stage or "grounded_draft",
            "model": kwargs.get("model"),
            "max_tokens": kwargs.get("max_tokens"),
        }
        started = time.perf_counter_ns()
        try:
            response = await original(self, *args, **kwargs)
        except Exception as exc:
            record.update(error=type(exc).__name__, status_code=getattr(exc, "status_code", None))
            raise
        else:
            record.update(_usage_dict(response), stop_reason=getattr(response, "stop_reason", None))
            return response
        finally:
            record["elapsed_ms"] = (time.perf_counter_ns() - started) // 1_000_000
            calls.append(record)

    return create


def install_llm_call_recorder() -> None:
    """Idempotently wrap the real Anthropic SDK; calls still go to the real provider."""
    global _installed
    if _installed:
        return
    from anthropic.resources.messages import AsyncMessages

    AsyncMessages.create = make_recording_create(AsyncMessages.create)  # type: ignore[method-assign]
    _installed = True


@contextmanager
def capture_llm_calls() -> Iterator[list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []
    token = _calls.set(calls)
    try:
        yield calls
    finally:
        _calls.reset(token)


# ---------------------------------------------------------------- trace summarising


def _find(trace: dict[str, Any], *path: str) -> Any:
    node: Any = trace
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def summarize_retrieval(trace: dict[str, Any]) -> dict[str, Any]:
    retrieval = trace.get("retrieval")
    if not isinstance(retrieval, dict):
        return {"used": False}
    hits = []
    cosines: list[float] = []
    for hit in retrieval.get("hits") or []:
        if not isinstance(hit, dict):
            continue
        cosine = hit.get("cosine")
        if isinstance(cosine, int | float):
            cosines.append(float(cosine))
        hits.append(
            {
                "id": hit.get("id"),
                "cosine": cosine,
                "score": hit.get("score") if hit.get("score") is not None else hit.get("rrf"),
                "heading": hit.get("heading") or hit.get("source_title"),
                "source_url": hit.get("source_url"),
            }
        )
    return {
        "used": retrieval.get("used", True),
        "mode": retrieval.get("mode"),
        "hit_count": len(hits),
        "best_cosine": max(cosines) if cosines else None,
        "hits": hits,
        "other": {k: v for k, v in retrieval.items() if k not in {"hits", "mode"}},
    }


def summarize_trace(raw: dict[str, Any], llm_calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Pull the stable, useful facts from the (JSON-safe) trace; unknown keys stay in raw."""
    errors: list[dict[str, Any]] = []
    for section, body in raw.items():
        if isinstance(body, dict):
            for key in ("error", "error_class", "error_detail"):
                if body.get(key):
                    errors.append({"section": section, key: body[key]})
    stage_timings = _find(raw, "decision", "stage_timings") or {}
    tokens = {
        name: sum(int(call.get(name) or 0) for call in llm_calls)
        for name in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }
    return {
        "stage_timings": stage_timings,
        "llm_call_count": len(llm_calls),
        "llm_calls": llm_calls,
        "tokens": tokens,
        "retrieval": summarize_retrieval(raw),
        "classification": raw.get("classification"),
        "understanding": raw.get("understand_turn"),
        "routing": raw.get("routing"),
        "validation": {
            key: raw[key]
            for key in raw
            if any(word in key for word in ("assess", "relevance", "repair", "validation"))
        },
        "provider": {
            k: v for k, v in (raw.get("provider") or {}).items() if k not in {"messages", "system"}
        }
        if isinstance(raw.get("provider"), dict)
        else raw.get("provider"),
        "decision": {k: v for k, v in (raw.get("decision") or {}).items() if k != "stage_timings"}
        if isinstance(raw.get("decision"), dict)
        else raw.get("decision"),
        "errors": errors,
    }


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]
