"""Regression tests for the first live-QA fix round (doubles only; no network).

Each test states a behaviour the live runs showed broken: the understanding call tolerating a
missing field, conversational intents never becoming knowledge gaps, the assistant answering
while a handoff is queued, identifiers never stored, emergencies sent to emergency services.
"""

from types import SimpleNamespace

import pytest
from sqlalchemy import select

from app.chat.outcome_copy import KEEP_HELPING_LINE, OUT_OF_SCOPE
from app.db import session_maker
from app.llm.safety_markers import has_identifier_mask, mask_identifiers
from app.llm.structured import structured_call
from app.llm.understanding import TurnUnderstanding
from app.models.handoff_context import HandoffContext
from app.models.handoff_outcome import HandoffOutcome
from app.services.bot_turn_service import without_site_name
from app.services.turn_routing import SiteProfile, wants_person
from tests.bot_fixtures import EASY_BODY, EASY_TITLE
from tests.fake_anthropic import Draft, FakeAnthropic, understanding
from tests.turn_helpers import open_chat

TIMING_CHUNK = (EASY_TITLE, EASY_BODY)
GOOD_ANSWER = Draft("Most negative results are reported within 24-48 hours.", EASY_BODY)


# ---- the understanding call is tolerant and retries once --------------------------------------


class _Scripted:
    """Anthropic-shaped client that returns scripted tool inputs, one per call."""

    def __init__(self, *inputs: dict) -> None:
        self.inputs = list(inputs)
        self.calls = 0
        self.messages = self

    async def create(self, **kwargs):
        self.calls += 1
        value = self.inputs.pop(0)
        return SimpleNamespace(
            stop_reason="tool_use",
            content=[SimpleNamespace(type="tool_use", name="understand_turn", input=value)],
        )


async def _understand(client) -> TurnUnderstanding:
    return await structured_call(
        client,
        name="understand_turn",
        schema=TurnUnderstanding,
        rules="r",
        payload={},
        model="m",
        max_tokens=100,
    )


async def test_missing_reply_is_not_a_failure(migrated_db) -> None:
    value = understanding()
    del value["reply"]
    result = await _understand(_Scripted(value))
    assert result.reply == ""


async def test_a_malformed_result_is_retried_once(migrated_db) -> None:
    client = _Scripted({"language": "en"}, understanding())
    result = await _understand(client)
    assert result.intent == "question"
    assert client.calls == 2


async def test_two_malformed_results_raise(migrated_db) -> None:
    client = _Scripted({"language": "en"}, {"language": "en"})
    with pytest.raises(ValueError):
        await _understand(client)
    assert client.calls == 2


# ---- conversational intents ---------------------------------------------------------------


async def test_greeting_uses_the_model_reply_even_when_the_site_name_has_digits(
    migrated_db,
) -> None:
    client = FakeAnthropic(
        understand={"intent": "smalltalk", "reply": "Hi! Welcome to 247 Flow, how can I help?"}
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, name="247 Flow", chunks=[TIMING_CHUNK])
        await chat.say("hi")
        reply = await chat.last_bot()
        hits = await chat.gap_hits()

    assert reply.body == "Hi! Welcome to 247 Flow, how can I help?"
    assert reply.response_reason_code == "smalltalk"
    assert hits == []
    assert client.stage_calls("draft") == []


@pytest.mark.parametrize(
    ("intent", "fallback"),
    [
        ("off_topic", OUT_OF_SCOPE),
        ("abuse", KEEP_HELPING_LINE),
        ("prompt_injection", KEEP_HELPING_LINE),
        ("smalltalk", KEEP_HELPING_LINE),
    ],
)
async def test_unusable_model_reply_for_a_conversational_intent_is_neutral_and_no_gap(
    migrated_db, intent, fallback
) -> None:
    client = FakeAnthropic(understand={"intent": intent, "reply": "See https://x.test"})
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("something")
        reply = await chat.last_bot()
        hits = await chat.gap_hits()

    assert reply.body == fallback
    assert hits == []
    assert client.stage_calls("draft") == []


async def test_a_gap_is_never_recorded_for_a_non_question_intent(migrated_db) -> None:
    client = FakeAnthropic(understand={"intent": "source_request", "reply": ""})
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("where did you get that?")
        assert await chat.gap_hits() == []


# ---- the unknown reply ----------------------------------------------------------------------


async def test_unknown_reply_is_written_by_its_own_call_and_offers_a_person(migrated_db) -> None:
    client = FakeAnthropic(
        understand={"unknown_reply": "No verified info on pricing. Want a person?"}
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("What does it cost?")
        reply = await chat.last_bot()

    assert reply.body == "No verified info on pricing. Want a person?"
    assert len(client.stage_calls("write_unknown_reply")) == 1
    assert reply.response_reason_code == "no_evidence"


async def test_accepting_the_offer_after_an_unknown_reply_opens_the_handoff(migrated_db) -> None:
    client = FakeAnthropic(
        understand=[{}, {"intent": "handoff_request", "reply": "I've asked a person to join."}]
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("What does it cost?")
        await chat.say("yes please")
        payload = client.stage_calls("understand_turn")[1]["messages"][0]["content"]
        handoffs = list((await session.scalars(select(HandoffContext))).all())

    assert '"previous_reply_offered_person": true' in payload
    assert len(handoffs) == 1
    assert chat.conversation.state == "queued"


# ---- the assistant keeps answering while a handoff is queued --------------------------------


async def test_assistant_answers_while_queued_and_a_withdrawal_stands_the_handoff_down(
    migrated_db,
) -> None:
    client = FakeAnthropic(
        understand=[
            {"intent": "handoff_request", "reply": "I've asked a person to join."},
            {"intent": "question", "query": "How quickly are results available?"},
            {
                "intent": "handoff_decline",
                "withdraws_handoff": True,
                "reply": "No problem, the request is withdrawn.",
            },
        ],
        draft=GOOD_ANSWER,
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("Can I talk to a person?")
        assert chat.conversation.state == "queued"
        assert chat.conversation.attention_needed is True

        await chat.say("How quickly are results available?")
        answer = await chat.last_bot()
        assert chat.conversation.state == "queued"
        assert answer.source_chunk_ids
        second = client.stage_calls("understand_turn")[1]["messages"][0]["content"]
        assert '"handoff_pending": true' in second

        await chat.say("never mind, no person")
        reply = await chat.last_bot()
        handoff = (await session.scalars(select(HandoffContext))).one()
        outcome = (await session.scalars(select(HandoffOutcome))).one()

    assert reply.body == "No problem, the request is withdrawn."
    assert chat.conversation.state == "bot"
    assert chat.conversation.attention_needed is False
    assert outcome.handoff_id == handoff.id
    assert outcome.outcome == "abandoned"
    assert outcome.resolved_by is None


async def test_repeated_request_while_queued_does_not_open_a_second_handoff(migrated_db) -> None:
    client = FakeAnthropic(
        understand=[
            {"intent": "handoff_request", "reply": "I've asked a person to join."},
            {"intent": "handoff_request", "reply": "A person was already asked."},
        ]
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("a person please")
        await chat.say("connect me now")
        reply = await chat.last_bot()
        handoffs = list((await session.scalars(select(HandoffContext))).all())

    assert reply.body == "A person was already asked."
    assert len(handoffs) == 1
    assert chat.conversation.state == "queued"


async def test_queued_unknown_reply_does_not_offer_another_person(migrated_db) -> None:
    client = FakeAnthropic(
        understand=[{"intent": "handoff_request", "reply": "Asked."}, {}],
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("a person please")
        await chat.say("What does it cost?")
    payload = client.stage_calls("write_unknown_reply")[0]["messages"][0]["content"]
    assert '"person_available": false' in payload
    assert '"handoff_pending": true' in payload


# ---- emergencies ------------------------------------------------------------------------------


async def test_urgent_safety_advises_emergency_services_and_only_offers_a_person(
    migrated_db,
) -> None:
    client = FakeAnthropic(
        understand={
            "intent": "urgent_safety",
            "reply": "Please contact local emergency services now. Shall I ask a person to join?",
        }
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("someone is hurt, it's urgent")
        reply = await chat.last_bot()
        handoffs = list((await session.scalars(select(HandoffContext))).all())
        hits = await chat.gap_hits()

    assert "emergency services" in reply.body
    assert reply.response_reason_code == "urgent_safety"
    assert handoffs == []
    assert hits == []
    assert chat.conversation.state == "bot"


# ---- routing details ----------------------------------------------------------------------------


def test_a_handoff_topic_cannot_match_when_the_site_has_none() -> None:
    parsed = TurnUnderstanding.model_validate(understanding(matches_handoff_topic=True))
    none = SiteProfile("S", "", (), True, (), "en", ())
    some = SiteProfile("S", "", ("legal advice",), True, (), "en", ())
    assert wants_person(parsed, none) is False
    assert wants_person(parsed, some) is True


def test_the_site_name_is_dropped_from_the_search_query_unless_the_visitor_wrote_it() -> None:
    assert without_site_name("onboarding for 247 Easy products", "247 Easy", "how to start") == (
        "onboarding for products"
    )
    assert (
        without_site_name("247 Easy pricing", "247 Easy", "does 247 Easy publish pricing")
        == "247 Easy pricing"
    )


# ---- identifiers ----------------------------------------------------------------------------------


def test_identifier_shapes_are_masked_but_contact_details_are_kept() -> None:
    text = "SSN 123-45-6789, born 1990-04-12, mail me at a@b.co or 555-123-4567"
    masked = mask_identifiers(text)
    assert "123-45-6789" not in masked
    assert "1990-04-12" not in masked
    assert "a@b.co" in masked
    assert "555-123-4567" in masked
    assert has_identifier_mask(masked)
    assert not has_identifier_mask(text)


async def test_shared_identifier_is_warned_about_without_a_handoff(migrated_db) -> None:
    client = FakeAnthropic(
        understand={
            "intent": "shared_identifiers",
            "reply": "Please don't share personal identifiers in chat; I removed it.",
        }
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("my SSN is 123-45-6789")
        rows = await chat.rows()
        handoffs = list((await session.scalars(select(HandoffContext))).all())

    assert [row.body for row in rows if row.role == "visitor"] == ["my SSN is [REDACTED]"]
    assert handoffs == []
    assert "identifiers" in (await chat.last_bot()).body


def test_a_personal_case_without_a_private_record_is_a_plain_question() -> None:
    bare = TurnUnderstanding.model_validate(understanding(intent="personal_case"))
    backed = TurnUnderstanding.model_validate(
        understanding(intent="personal_case", own_record_quote="my result")
    )
    assert bare.intent == "question"
    assert backed.intent == "personal_case"
