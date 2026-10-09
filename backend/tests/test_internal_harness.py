"""TEMPORARY chatbot harness: gating, localhost-only access, and the endpoint contract.

Provider doubles are test-only; the harness itself never substitutes providers.
"""

import asyncio
from types import SimpleNamespace

from fastapi.testclient import TestClient

import app.main as main_module
from app.db import reset_engine
from app.llm.bot_responder import BotResponder
from app.models.user import User
from app.security.deps import get_current_admin
from app.services import harness_store
from app.services.harness_store import HarnessStore
from app.services.harness_telemetry import (
    capture_llm_calls,
    json_safe,
    make_recording_create,
)
from app.settings import reset_settings_cache
from tests.test_internal_dev_chat import SERVICE_REPLY, _GroundedClient, _seed_workbench

BASE = "/api/internal/dev/harness"


def _client(monkeypatch, *, flag: bool, host: str = "127.0.0.1") -> TestClient:
    monkeypatch.setenv("APP_ENV", "local")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-for-fake-embedder")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-key-for-grounded-client")
    if flag:
        monkeypatch.setenv("INTERNAL_DEV_HARNESS", "1")
    else:
        monkeypatch.delenv("INTERNAL_DEV_HARNESS", raising=False)
    reset_settings_cache()
    monkeypatch.setattr("app.llm.bot_responder.AsyncAnthropic", _GroundedClient)
    BotResponder._shared_client = None
    BotResponder._shared_client_factory = None
    asyncio.run(_seed_workbench())
    reset_engine()
    monkeypatch.setattr(harness_store, "_store", HarnessStore(log_path=None))
    application = main_module.create_app()
    application.dependency_overrides[get_current_admin] = lambda: User(is_admin=True)
    return TestClient(application, client=(host, 50000))


def test_disabled_without_flag(migrated_db, monkeypatch) -> None:
    with _client(monkeypatch, flag=False) as client:
        assert client.get(f"{BASE}/stats").status_code == 404
        assert client.get(f"{BASE}/sites").status_code == 404


def test_refused_from_non_loopback_peer(migrated_db, monkeypatch) -> None:
    with _client(monkeypatch, flag=True, host="203.0.113.9") as client:
        assert client.get(f"{BASE}/stats").status_code == 404
        assert (
            client.post(f"{BASE}/conversations", json={"site_key": "samplesite"}).status_code == 404
        )


def test_requires_staff_auth(migrated_db, monkeypatch) -> None:
    with _client(monkeypatch, flag=True) as client:
        client.app.dependency_overrides.clear()
        assert client.get(f"{BASE}/stats").status_code == 401


def test_full_flow(migrated_db, monkeypatch) -> None:
    with _client(monkeypatch, flag=True) as client:
        sites = client.get(f"{BASE}/sites").json()
        site = next(s for s in sites if s["key"] == "samplesite")
        assert site["bot_enabled"] is True and site["live_chunks"] >= 2
        assert "Our services" in site["sample_headings"]

        created = client.post(f"{BASE}/conversations", json={"site_key": "samplesite"})
        assert created.status_code == 200, created.text
        conv = created.json()["conversation_id"]
        assert created.json()["state"] == "bot"

        turn = client.post(
            f"{BASE}/conversations/{conv}/messages", json={"text": "what services do you provide?"}
        )
        assert turn.status_code == 200, turn.text
        data = turn.json()
        assert SERVICE_REPLY in data["reply_text"]
        assert data["state"] == "bot"
        assert data["citations"] and all(c["belongs_to_site"] for c in data["citations"])
        assert data["citations"][0]["cited_text"]
        assert data["handoff_opened"] is False
        assert data["latency_ms"] >= 0
        assert "stage_timings" in data["telemetry"]
        assert isinstance(data["trace"], dict) and "decision" in data["trace"]

        shown = client.get(f"{BASE}/conversations/{conv}").json()
        assert len(shown["turns"]) == 1
        assert [m["role"] for m in shown["messages"]][-2:] == ["visitor", "bot"]

        stats = client.get(f"{BASE}/stats").json()
        assert stats["turns"] == 1
        assert stats["foreign_citation_count"] == 0
        assert stats["by_site"] == {"samplesite": 1}
        assert stats["citation_rate"] == 1.0

        assert client.post(f"{BASE}/conversations", json={"site_key": "nope"}).status_code == 404
        missing = "00000000-0000-0000-0000-000000000000"
        assert client.get(f"{BASE}/conversations/{missing}").status_code == 404


def test_json_safe_redacts_and_handles_odd_types() -> None:
    import uuid
    from dataclasses import dataclass

    @dataclass
    class Hit:
        id: uuid.UUID
        embedding: list[float]
        note: str

    out = json_safe({"hits": [Hit(uuid.uuid4(), [0.1], "mail me at a.b@example.com")], "x": {1, 2}})
    assert "embedding" not in out["hits"][0]
    assert "a.b@example.com" not in out["hits"][0]["note"]
    assert out["x"] == [1, 2]


def test_recording_create_notes_usage_and_passes_through() -> None:
    async def original(self, **kwargs):
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=3,
                cache_read_input_tokens=7,
                cache_creation_input_tokens=0,
            ),
            stop_reason="end_turn",
        )

    wrapped = make_recording_create(original)

    async def run():
        with capture_llm_calls() as calls:
            await wrapped(
                None,
                model="m",
                max_tokens=5,
                tool_choice={"type": "tool", "name": "understand_turn"},
            )
        return calls

    calls = asyncio.run(run())
    assert calls[0]["stage"] == "understand_turn" and calls[0]["cache_read_input_tokens"] == 7
    assert asyncio.run(wrapped(None, model="m")).stop_reason == "end_turn"


def test_stats_empty_and_counts_foreign() -> None:
    store = HarnessStore(log_path=None)
    assert store.stats() == {"turns": 0}
    store.add(
        {
            "conversation_id": "c",
            "site_key": "s",
            "intent": None,
            "outcome": "x",
            "reason_code": "no_answer",
            "latency_ms": 10,
            "citations": [{"belongs_to_site": False}],
            "handoff_opened": False,
            "knowledge_gap_recorded": True,
            "telemetry": {"errors": [], "llm_call_count": 2, "tokens": {"input_tokens": 4}},
        }
    )
    stats = store.stats()
    assert stats["foreign_citation_count"] == 1 and stats["unknown_or_abstain_rate"] == 1.0
