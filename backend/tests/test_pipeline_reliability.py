"""Reliability, cost and correctness of the Claude pipeline around one bot turn."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from anthropic.types import Message as ProviderMessage
from sqlalchemy import select

from app.chat.outcome_copy import RATE_CEILING_HUMAN, TECH_FAIL_HUMAN, TECH_FAIL_SOLO
from app.db import session_maker
from app.llm.bot_responder import BotResponder
from app.llm.understanding import TurnUnderstanding, UnderstandingInput, understand_turn
from app.models.message import Message
from app.services.conversation_service import ConversationService
from app.services.full_context import conversation_window_messages
from app.services.grounded_response import (
    Citation,
    EvidenceUnit,
    GroundedResponseEngine,
    ModelDraft,
    TurnContext,
    unknown_decision,
)
from app.services.rate_ceiling import TurnBudget
from app.services.rate_limit import RateLimitExceeded
from app.settings import get_settings, reset_settings_cache
from tests.bot_fixtures import EASY_BODY, EASY_TITLE, insert_bot_conversation, insert_site
from tests.fake_anthropic import Draft, FakeAnthropic, understanding
from tests.turn_helpers import open_chat
from tests.ws_helpers import HOST_ORIGIN


def unit(text: str) -> EvidenceUnit:
    return EvidenceUnit(
        id=uuid4(),
        canonical_question=None,
        aliases=(),
        topic_label="Sales",
        answer_verbatim=text,
        source_title="Sales",
        source_url="https://example.test/sales",
        snapshot_id=uuid4(),
    )


async def respond(body: str, item: EvidenceUnit, claim: str, **turn_kwargs):
    start = body.index(claim)
    citation = Citation(
        item.id,
        item.snapshot_id,
        start,
        start + len(claim),
        0,
        len(item.answer_verbatim),
        item.answer_verbatim,
        item.source_title,
        item.source_url,
    )

    async def complete(_turn, _units):
        return ModelDraft(body, [citation])

    return await GroundedResponseEngine(complete).respond(
        TurnContext("What is offered?", [item], **turn_kwargs)
    )


def understood(text: str, **overrides) -> TurnUnderstanding:
    return TurnUnderstanding.model_validate(understanding(query=text, **overrides))


# --- validator limits ---------------------------------------------------------


async def test_answer_over_120_words_is_kept_when_within_the_character_ceiling():
    claim = " ".join(f"word{index}" for index in range(140)) + "."
    item = unit(claim)
    result = await respond("Yes. " + claim, item, claim)
    assert result.reason_code != "grounding_reject"
    assert len(result.body.split()) > 120


async def test_answer_over_character_ceiling_is_rejected(monkeypatch):
    claim = ("alpha " * 300).strip() + "."  # 1800 characters
    item = unit(claim)
    result = await respond("Yes. " + claim, item, claim, understanding=understood("q"))
    assert result.reason_code == "grounding_reject"
    # The ceiling is the configured setting, not a hard-coded 1500.
    monkeypatch.setenv("MAX_BOT_ANSWER_CHARS", "2500")
    reset_settings_cache()
    result = await respond("Yes. " + claim, item, claim)
    assert result.reason_code != "grounding_reject"


# --- human_enabled ------------------------------------------------------------


async def test_failed_provider_without_staff_never_promises_a_specialist():
    async def complete(_turn, _units):
        return None

    item = unit("Email sales@example.com.")
    engine = GroundedResponseEngine(complete)
    solo = await engine.respond(TurnContext("price?", [item], human_enabled=False))
    assert solo.body == TECH_FAIL_SOLO
    assert solo.offer_handoff is False
    staffed = await engine.respond(TurnContext("price?", [item]))
    assert staffed.body == TECH_FAIL_HUMAN
    assert staffed.offer_handoff is True


async def test_unverifiable_answer_without_staff_offers_no_person_and_shows_contacts():
    item = unit("Email sales@example.com.")

    turn = TurnContext(
        "price?",
        [item],
        human_enabled=False,
        contact_info=("sales@example.com",),
        understanding=understood("price?"),
    )
    decision = unknown_decision(turn, text="I can't confirm that. Our team can help directly.")
    assert decision.body == "I can't confirm that. Our team can help directly.\n\nsales@example.com"
    assert decision.offer_handoff is False


async def test_run_bot_turn_exception_path_respects_site_human_setting(migrated_db):
    async with session_maker()() as session:
        site = await insert_site(session, "solo-" + uuid4().hex[:8], "Solo Brand")
        site.human_enabled = False
        visitor, conversation = await insert_bot_conversation(session, site)
        await session.commit()
        service = ConversationService(session)
        submitted = await service.visitor_message(
            conversation.id, visitor.id, HOST_ORIGIN, uuid4(), "What does it cost?"
        )

        async def boom(*_args, **_kwargs):
            raise RuntimeError("provider down")

        service._bot_turn.prepare = boom
        await service.run_bot_turn(conversation.id, submitted.generation_id)
        body = await session.scalar(
            select(Message.body).where(
                Message.conversation_id == conversation.id, Message.role == "bot"
            )
        )
    assert body == TECH_FAIL_SOLO


# --- staff attribution --------------------------------------------------------


def test_staff_messages_are_marked_so_they_are_not_the_bots_own_claims():
    rows = [
        Message(role="visitor", body="Can I get a discount?"),
        Message(role="agent", body="Yes, 20% off for new accounts."),
        Message(role="bot", body="Please email sales."),
    ]
    messages = conversation_window_messages(rows)
    assert messages == [
        {"role": "user", "content": "Can I get a discount?"},
        {"role": "assistant", "content": "[Staff member] Yes, 20% off for new accounts."},
        {"role": "assistant", "content": "Please email sales."},
    ]


# --- retries and degradation --------------------------------------------------


async def test_shared_client_retries_transient_errors_within_the_turn_deadline(monkeypatch):
    seen: dict = {}

    class Client:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        async def close(self):
            pass

    await BotResponder.close_shared_client()
    monkeypatch.setattr("app.llm.bot_responder.AsyncAnthropic", Client)
    try:
        BotResponder._shared_anthropic_client()
    finally:
        await BotResponder.close_shared_client()
    settings = get_settings()
    assert seen["max_retries"] == 2
    # Worst case for one call (every attempt times out) must leave room in the turn lease.
    assert (
        seen["timeout"] * (seen["max_retries"] + 1) <= settings.bot_generation_lease_seconds * 0.75
    )


async def test_assessor_failure_keeps_a_plain_standalone_cited_answer(migrated_db):
    client = FakeAnthropic(
        draft=Draft("Most negative results are reported within 24-48 hours.", EASY_BODY),
        fail={"assess_answer"},
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[(EASY_TITLE, EASY_BODY)])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()
    # A plain, fully cited, standalone answer in the site's language needs no assessor at all.
    assert reply.source_chunk_ids
    assert reply.response_reason_code is None


async def test_assessor_failure_is_released_only_for_plain_answers_when_assessment_is_forced(
    migrated_db, monkeypatch
):
    monkeypatch.setenv("ANSWER_ASSESSOR_MODE", "always")
    client = FakeAnthropic(
        draft=Draft("Most negative results are reported within 24-48 hours.", EASY_BODY),
        fail={"assess_answer"},
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[(EASY_TITLE, EASY_BODY)])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()
    assert reply.source_chunk_ids
    assert len(client.stage_calls("assess_answer")) == 1


# --- per-conversation Claude budget -------------------------------------------


def test_default_budget_covers_several_full_turns():
    assert get_settings().anthropic_calls_per_minute == 30


async def test_turn_budget_allows_exactly_the_configured_calls(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_CALLS_PER_MINUTE", "3")
    reset_settings_cache()
    budget = TurnBudget(uuid4())
    for _ in range(3):
        await budget.charge()
    with pytest.raises(RateLimitExceeded):
        await budget.charge()
    # Another conversation has its own budget.
    await TurnBudget(uuid4()).charge()


async def test_ceiling_stops_claude_calls_and_replies_with_the_rate_ceiling_copy(
    migrated_db, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_CALLS_PER_MINUTE", "1")
    reset_settings_cache()
    client = FakeAnthropic(draft=Draft("never", None))
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[(EASY_TITLE, EASY_BODY)])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()
        await session.refresh(chat.conversation)

    assert len(client.stage_calls("understand_turn")) == 1
    assert client.stage_calls("draft") == []
    assert reply.body == RATE_CEILING_HUMAN
    assert reply.response_reason_code == "rate_ceiling"
    assert chat.conversation.state == "bot"


# --- prompt layout and citations ----------------------------------------------


class _CapturingClient:
    def __init__(self, captured: list[dict], pick: int):
        self.captured, self.pick = captured, pick
        self.messages = self

    async def close(self):
        pass

    async def create(self, **kwargs):
        self.captured.append(kwargs)
        documents = [
            block
            for message in kwargs["messages"]
            if isinstance(message["content"], list)
            for block in message["content"]
            if block.get("type") == "document"
        ]
        document = documents[self.pick]
        source = document["source"]["data"]
        return ProviderMessage.model_validate(
            {
                "id": "msg_layout",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-4-5",
                "content": [
                    {
                        "type": "text",
                        "text": "Answer.",
                        "citations": [
                            {
                                "type": "char_location",
                                "document_index": self.pick,
                                "document_title": document["title"],
                                "start_char_index": 0,
                                "end_char_index": len(source),
                                "cited_text": source,
                            }
                        ],
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
        )


async def test_history_is_the_cached_prefix_and_documents_ride_with_the_new_question():
    captured: list[dict] = []
    first, second = unit("First passage."), unit("Second passage.")
    history = (
        {"role": "user", "content": "Hello there"},
        {"role": "assistant", "content": "Hi, how can I help?"},
        {"role": "user", "content": "Do you screen drivers?"},
        {"role": "assistant", "content": "Yes we do."},
    )
    turn = TurnContext(
        "What does it cost?",
        [first, second],
        "Acme",
        prior_messages=history,
        understanding=understood("cost", language="es"),
    )
    draft = await BotResponder(client=_CapturingClient(captured, pick=1)).generate_grounded_draft(
        turn, [first, second]
    )

    messages = captured[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    # The stable prefix ends with a cache breakpoint on the last history message.
    prefix = messages[3]["content"]
    assert prefix == [
        {"type": "text", "text": "Yes we do.", "cache_control": {"type": "ephemeral"}}
    ]
    assert "cache_control" not in str(messages[:3])
    # Per-turn evidence is not cached: it changes every turn.
    final = messages[4]["content"]
    assert [block["type"] for block in final] == ["document", "document", "text"]
    assert all("cache_control" not in block for block in final)
    assert "What does it cost?" in final[2]["text"]
    assert '"language": "es"' in final[2]["text"]
    # document_index counts documents in request order, so index 1 is the second unit.
    assert [c.chunk_id for c in draft.citations] == [second.id]
    assert captured[0]["max_tokens"] == get_settings().anthropic_max_tokens


async def test_history_starting_with_assistant_still_sends_a_user_first_request():
    captured: list[dict] = []
    only = unit("Only passage.")
    turn = TurnContext(
        "Price?",
        [only],
        "Acme",
        prior_messages=({"role": "assistant", "content": "[Staff member] Welcome back."},),
    )
    await BotResponder(client=_CapturingClient(captured, pick=0)).generate_grounded_draft(
        turn, [only]
    )
    roles = [m["role"] for m in captured[0]["messages"]]
    assert roles[0] == "user"


async def test_site_scope_reaches_the_drafting_system_prompt():
    captured: list[dict] = []
    only = unit("Only passage.")
    turn = TurnContext("Price?", [only], "Acme", assistant_scope="lab testing and compliance")
    await BotResponder(client=_CapturingClient(captured, pick=0)).generate_grounded_draft(
        turn, [only]
    )
    system = captured[0]["system"][0]["text"]
    assert "Acme" in system
    assert "lab testing and compliance" in system


async def test_structured_calls_have_room_for_a_full_instruction():
    seen: dict = {}

    class Messages:
        async def create(self, **kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                stop_reason="tool_use",
                content=[
                    SimpleNamespace(
                        type="tool_use", name="understand_turn", input=understanding(query="q")
                    )
                ],
            )

    inputs = UnderstandingInput(
        site_name="Acme",
        primary_language="en",
        assistant_scope="",
        knowledge_topics=(),
        handoff_topics=(),
        human_enabled=True,
        contact_available=False,
        history=(),
        latest_message="hi",
    )
    await understand_turn(SimpleNamespace(messages=Messages()), inputs)
    assert seen["max_tokens"] == get_settings().understanding_max_tokens
    assert seen["tool_choice"] == {"type": "tool", "name": "understand_turn"}


async def test_repair_budget_starts_when_repair_begins(monkeypatch) -> None:
    from app.services import bot_turn_service
    from app.services.bot_turn_service import BotTurnService

    monkeypatch.setattr(
        bot_turn_service,
        "get_settings",
        lambda: SimpleNamespace(anthropic_timeout=0.2),
    )
    calls: list[str] = []

    class Responder:
        async def generate_grounded_draft(self, *_a, **_k):
            return None

        async def repair_grounded_draft(self, _turn, _units, draft, failure, **_k):
            calls.append((draft, failure))
            return "repaired"

    engine = BotTurnService(None, responder=Responder(), embedder=object())._engine({})  # type: ignore[arg-type]
    # A slow (retried) draft eats more than one provider timeout before repair starts.
    await asyncio.sleep(0.3)
    assert await engine._repair(None, [], "draft", "no_citation") == "repaired"
    assert calls == [("draft", "no_citation")]
