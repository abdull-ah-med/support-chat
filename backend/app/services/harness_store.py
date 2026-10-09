"""TEMPORARY: in-memory per-turn store, aggregate stats and JSONL log for the chat harness."""

from __future__ import annotations

import json
import threading
from collections import Counter
from pathlib import Path
from typing import Any

from app.services.harness_telemetry import percentile

LOG_PATH = Path(__file__).resolve().parents[2] / ".harness" / "turns.jsonl"
ABSTAIN_WORDS = ("unknown", "no_answer", "abstain", "no_evidence", "below_relevance", "fallback")


class HarnessStore:
    def __init__(self, log_path: Path | None = LOG_PATH) -> None:
        self._turns: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._log_path = log_path

    def add(self, turn: dict[str, Any]) -> None:
        with self._lock:
            self._turns.append(turn)
            if self._log_path is not None:
                try:
                    self._log_path.parent.mkdir(parents=True, exist_ok=True)
                    with self._log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(turn, ensure_ascii=False, default=str) + "\n")
                except OSError:
                    pass

    def for_conversation(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [t for t in self._turns if t["conversation_id"] == conversation_id]

    def known(self, conversation_id: str) -> bool:
        return bool(self.for_conversation(conversation_id))

    def stats(self) -> dict[str, Any]:
        with self._lock:
            turns = list(self._turns)
        n = len(turns)
        if not n:
            return {"turns": 0}

        def rate(count: int) -> float:
            return round(count / n, 4)

        latencies = [float(t["latency_ms"]) for t in turns]
        reasons = [str(t.get("reason_code") or "") for t in turns]
        abstain = sum(
            1
            for t, r in zip(turns, reasons, strict=True)
            if any(w in r for w in ABSTAIN_WORDS) or t.get("knowledge_gap_recorded")
        )
        return {
            "turns": n,
            "by_site": dict(Counter(t["site_key"] for t in turns)),
            "by_intent": dict(Counter(str(t.get("intent")) for t in turns)),
            "by_outcome": dict(Counter(str(t.get("outcome")) for t in turns)),
            "by_reason": dict(Counter(reasons)),
            "unknown_or_abstain_rate": rate(abstain),
            "handoff_rate": rate(sum(1 for t in turns if t.get("handoff_opened"))),
            "citation_rate": rate(sum(1 for t in turns if t.get("citations"))),
            "foreign_citation_count": sum(
                1 for t in turns for c in t.get("citations") or [] if not c["belongs_to_site"]
            ),
            "error_count": sum(1 for t in turns if t["telemetry"].get("errors") or t.get("error")),
            "latency_ms": {
                "p50": percentile(latencies, 50),
                "p95": percentile(latencies, 95),
                "max": max(latencies),
            },
            "llm_calls_per_turn_avg": round(
                sum(t["telemetry"]["llm_call_count"] for t in turns) / n, 2
            ),
            "tokens_per_turn_avg": {
                name: round(sum(t["telemetry"]["tokens"][name] for t in turns) / n, 1)
                for name in turns[0]["telemetry"]["tokens"]
            },
        }


_store: HarnessStore | None = None


def get_store() -> HarnessStore:
    global _store
    if _store is None:
        _store = HarnessStore()
    return _store
