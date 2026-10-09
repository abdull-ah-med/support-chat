"""Semantic review of drafted answers: accept, trim, repair once, or say "unknown".

Citations prove a passage was quoted; the assessor decides whether the answer serves the question.
Provider replies are scripted, so each test states the review outcome it is exercising.
"""

import json

from app.db import session_maker
from tests.fake_anthropic import UNKNOWN_REPLY, Draft, FakeAnthropic, assessment
from tests.turn_helpers import open_chat

WORKFLOW = (
    "Candidates receive an authorization and choose a nearby collection site, "
    "usually without an appointment."
)
CONTACT = "To discuss pricing with a specialist, email inquiries@example.test."
CHUNKS = [
    ("Candidate collection appointment", WORKFLOW),
    ("Specialist pricing contact", CONTACT),
]


async def _ask(client, text="how do i set up an appointment", **fields):
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=CHUNKS, **fields)
        await chat.say(text)
        reply = await chat.last_bot()
        await session.refresh(chat.conversation)
        citations = [c.cited_text for c in reply.citations]
    return chat, reply, citations


async def test_ambiguity_is_a_question_not_an_assumed_workflow(migrated_db) -> None:
    client = FakeAnthropic(
        understand={
            "ambiguity": "arranging a sales call or booking a collection visit",
            "reply": "Do you want a call with the team, or a collection visit?",
        },
        draft=Draft(WORKFLOW, WORKFLOW),
    )
    _chat, reply, _ = await _ask(client)

    assert reply.body == "Do you want a call with the team, or a collection visit?"
    assert reply.response_outcome == "clarification"
    assert client.stage_calls("draft") == []


async def test_answer_for_the_wrong_workflow_is_repaired_once_then_unknown(migrated_db) -> None:
    reject = assessment("reject", "wrong_action", repair_instruction="wrong actor")
    client = FakeAnthropic(
        understand={"relation": "correction", "query": "Specialist pricing contact"},
        draft=[Draft(CONTACT, CONTACT), Draft(CONTACT, CONTACT)],
        assess=[reject, reject],
    )
    _chat, reply, citations = await _ask(client, "i meant with the specialist")

    assert reply.body.startswith(UNKNOWN_REPLY)
    assert reply.response_reason_code == "grounding_reject"
    assert citations == []
    assert len(client.stage_calls("repair")) == 1
    repair = client.stage_calls("repair")[0]
    assert json.loads(repair["messages"][-1]["content"])["failure"] == "wrong actor"


async def test_repair_that_fixes_the_answer_is_reassessed_and_kept(migrated_db) -> None:
    reject = assessment("reject", "wrong_subject", repair_instruction="use the contact passage")
    client = FakeAnthropic(
        understand={"relation": "continuation", "query": "Specialist pricing contact"},
        draft=[
            Draft("Reach out to the team about pricing, via inquiries@example.test.", CONTACT),
            Draft(CONTACT, CONTACT),
        ],
        assess=[reject, assessment("supported_next_step")],
    )
    _chat, reply, citations = await _ask(client, "i meant with the specialist")

    assert reply.body == CONTACT
    assert reply.response_reason_code == "supported_next_step"
    assert reply.response_outcome == "partial_answer"
    assert citations == [CONTACT]


async def test_assessor_can_remove_one_unsupported_sentence_and_keep_the_citations(
    migrated_db,
) -> None:
    promise = " A specialist will call you tomorrow."
    client = FakeAnthropic(
        understand={"relation": "continuation", "query": "Specialist pricing contact"},
        draft=Draft(CONTACT, CONTACT, tail=promise),
        assess=[
            assessment("reject", "unsupported_claim", remove_sentences=[promise.strip()]),
            assessment("supported_next_step"),
        ],
    )
    _chat, reply, citations = await _ask(client, "can a specialist call me")

    assert reply.body == CONTACT
    assert citations == [CONTACT]
    assert client.stage_calls("repair") == []
    assert len(client.stage_calls("assess_answer")) == 2


async def test_no_evidence_gives_the_models_precise_limitation_not_a_redirect(migrated_db) -> None:
    client = FakeAnthropic(
        understand={"unknown_reply": "I have no verified information on weekend appointments."}
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("can I book on weekends?")
        reply = await chat.last_bot()

    assert reply.body.startswith("I have no verified information on weekend appointments.")
    assert reply.response_reason_code == "no_evidence"
    assert client.stage_calls("draft") == []


async def test_public_contact_without_provider_attribution_is_verified_and_persisted(
    migrated_db,
) -> None:
    # The model wrote the address but cited only the lead-in; the address occurs verbatim in the
    # passage, so it is attributed to it instead of being rejected as an unsupported contact.
    client = FakeAnthropic(
        understand={"relation": "continuation", "query": "Specialist pricing contact"},
        draft=Draft(
            "To discuss pricing, email",
            "To discuss pricing with a specialist,",
            tail=" inquiries@example.test.",
        ),
        assess=assessment("supported_next_step"),
    )
    _chat, reply, citations = await _ask(client, "how do I reach sales about pricing")

    assert reply.body == "To discuss pricing, email inquiries@example.test."
    assert any("inquiries@example.test" in text for text in citations)


async def test_contact_not_in_the_knowledge_is_never_released(migrated_db) -> None:
    client = FakeAnthropic(
        understand={"relation": "continuation", "query": "Specialist pricing contact"},
        draft=Draft("Email", "To discuss pricing with a specialist,", tail=" nobody@example.test."),
        assess=assessment("supported_next_step"),
    )
    _chat, reply, _citations = await _ask(client, "how do I reach sales about pricing")

    assert "nobody@example.test" not in reply.body
    assert reply.body.startswith(UNKNOWN_REPLY)


async def test_question_about_what_the_chat_itself_did_is_answered_from_the_conversation(
    migrated_db,
) -> None:
    client = FakeAnthropic(
        understand={
            "intent": "conversation_recap",
            "reply": "No, I haven't booked anything, and I can't from this chat.",
        }
    )
    _chat, reply, _ = await _ask(client, "did you book my appointment?")

    assert reply.body == "No, I haven't booked anything, and I can't from this chat."
    assert reply.source_chunk_ids is None
    assert client.stage_calls("draft") == []


async def test_assessor_supported_next_step_is_a_partial_answer_without_a_miss(migrated_db) -> None:
    client = FakeAnthropic(
        understand={"relation": "continuation", "query": "Specialist pricing contact"},
        draft=Draft(CONTACT, CONTACT),
        assess=assessment("supported_next_step"),
    )
    chat, reply, _ = await _ask(client, "what does it cost")

    assert reply.response_outcome == "partial_answer"
    assert chat.conversation.fallback_count == 0


async def test_assessor_unsupported_detail_counts_as_a_gap(migrated_db) -> None:
    client = FakeAnthropic(
        understand={"relation": "continuation", "query": "Specialist pricing contact"},
        draft=Draft(CONTACT, CONTACT),
        assess=assessment("unsupported_detail"),
    )
    chat, reply, _ = await _ask(client, "what does it cost")

    assert reply.response_reason_code == "needs_confirmation"
    assert chat.conversation.fallback_count == 1
    assert len(await chat.gap_hits()) == 1
