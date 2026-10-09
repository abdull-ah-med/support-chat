"""The turn flow end to end: one understanding call routes the turn, knowledge answers it.

Every provider call goes to a fake Anthropic client; nothing here reaches the network. Oracles
are stated behaviours: the bot never drafts without relevant evidence, never invents text when
a provider fails, mirrors the visitor's language, and takes its per-site behaviour from the
site's configuration.
"""

import json

from app.chat.outcome_copy import CALLBACK_LINE, TECH_FAIL_HUMAN, TECH_FAIL_SOLO
from app.db import session_maker
from app.models.canned_reply import CannedReply
from app.models.conversation import Conversation
from app.settings import reset_settings_cache
from tests.bot_fixtures import EASY_BODY, EASY_TITLE
from tests.fake_anthropic import (
    NO_ANSWER_DRAFT,
    UNKNOWN_REPLY,
    Draft,
    FakeAnthropic,
    assessment,
)
from tests.turn_helpers import open_chat

TIMING_CHUNK = (EASY_TITLE, EASY_BODY)
GOOD_ANSWER = Draft("Most negative results are reported within 24-48 hours.", EASY_BODY)


def understood(client: FakeAnthropic) -> dict:
    """The payload the model received for its first understanding call."""
    return json.loads(client.stage_calls("understand_turn")[0]["messages"][0]["content"])


# ---- don't generate when there is nothing to say ------------------------------------------


async def test_irrelevant_evidence_skips_the_drafting_model_and_records_a_gap(migrated_db) -> None:
    # "Are results reported?" matches the passage by words, but its embedding is unrelated.
    client = FakeAnthropic(understand={"query": "Are results reported?"})
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("Are results reported?")
        reply = await chat.last_bot()
        hits = await chat.gap_hits()

    assert client.stage_calls("draft") == []
    assert reply.body.startswith(UNKNOWN_REPLY)
    assert reply.response_reason_code == "no_evidence"
    assert reply.response_outcome == "knowledge_gap"
    assert reply.source_chunk_ids is None
    assert len(hits) == 1


async def test_no_knowledge_at_all_replies_unknown_without_any_draft_call(migrated_db) -> None:
    client = FakeAnthropic()
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()

    assert client.stage_calls("draft") == []
    assert reply.body.startswith(UNKNOWN_REPLY)
    assert reply.response_reason_code == "no_evidence"


async def test_no_answer_sentinel_from_the_drafting_model_takes_the_unknown_path(
    migrated_db,
) -> None:
    client = FakeAnthropic(draft=NO_ANSWER_DRAFT)
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()
        hits = await chat.gap_hits()

    assert len(client.stage_calls("draft")) == 1
    assert reply.body.startswith(UNKNOWN_REPLY)
    assert "<no_answer" not in reply.body
    assert reply.response_reason_code == "no_evidence"
    assert len(hits) == 1


async def test_unknown_reply_offers_a_person_when_one_can_join(migrated_db) -> None:
    client = FakeAnthropic(
        understand={"unknown_reply": "I have nothing verified on that. Shall I connect you?"}
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("What is the weather?")
        reply = await chat.last_bot()
        await session.refresh(chat.conversation)
        offered = chat.conversation.fallback_count

    assert reply.body == "I have nothing verified on that. Shall I connect you?"
    assert offered >= 1
    assert understood(client)["site"]["human_enabled"] is True


async def test_unknown_reply_shows_the_configured_contact_route_when_no_person_can_join(
    migrated_db,
) -> None:
    client = FakeAnthropic()
    async with session_maker()() as session:
        chat = await open_chat(
            session,
            client,
            human_enabled=False,
            contact_info=["help@example.test", "555-0100"],
        )
        await chat.say("What is the weather?")
        reply = await chat.last_bot()

    assert reply.body == f"{UNKNOWN_REPLY}\n\nhelp@example.test\n555-0100"
    assert understood(client)["site"]["contact_available"] is True


# ---- answering ---------------------------------------------------------------------------


async def test_every_answer_is_assessed_by_default(migrated_db) -> None:
    client = FakeAnthropic(draft=GOOD_ANSWER)
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")

    assert len(client.stage_calls("assess_answer")) == 1


async def test_plain_cited_answer_in_the_site_language_skips_the_assessor_in_risk_mode(
    migrated_db, monkeypatch
) -> None:
    monkeypatch.setenv("ANSWER_ASSESSOR_MODE", "risk_based")
    reset_settings_cache()
    client = FakeAnthropic(draft=GOOD_ANSWER)
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()
        await session.refresh(chat.conversation)

    assert reply.body == GOOD_ANSWER.text
    assert reply.source_chunk_ids
    assert client.stage_calls("assess_answer") == []
    assert len(client.stage_calls("understand_turn")) == 1
    assert chat.conversation.intent == "question"


async def test_continuation_is_always_assessed(migrated_db) -> None:
    client = FakeAnthropic(draft=GOOD_ANSWER, understand={"relation": "continuation"})
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")

    assert len(client.stage_calls("assess_answer")) == 1


async def test_assessor_mode_always_assesses_plain_answers(migrated_db, monkeypatch) -> None:
    monkeypatch.setenv("ANSWER_ASSESSOR_MODE", "always")
    client = FakeAnthropic(draft=GOOD_ANSWER)
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")

    assert len(client.stage_calls("assess_answer")) == 1


async def test_partial_answer_with_an_uncited_sentence_is_assessed(migrated_db) -> None:
    client = FakeAnthropic(
        draft=Draft(
            "Most negative results are reported within 24-48 hours.",
            EASY_BODY,
            tail=" I can't confirm anything else.",
        )
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")

    assert len(client.stage_calls("assess_answer")) == 1


async def test_unsupported_number_is_repaired_once_then_answered_unknown(migrated_db) -> None:
    wrong = Draft("Results are reported within 99 hours.", EASY_BODY)
    client = FakeAnthropic(draft=[wrong, wrong])
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()

    assert len(client.stage_calls("draft")) == 1
    assert len(client.stage_calls("repair")) == 1
    assert reply.body.startswith(UNKNOWN_REPLY)
    assert reply.response_reason_code == "grounding_reject"


async def test_assessor_rejection_gets_one_repair_then_the_unknown_path(migrated_db) -> None:
    reject = assessment("reject", "unsupported_claim", repair_instruction="drop the claim")
    client = FakeAnthropic(
        draft=[GOOD_ANSWER, GOOD_ANSWER],
        understand={"relation": "continuation"},
        assess=[reject, reject],
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()

    assert len(client.stage_calls("repair")) == 1
    assert len(client.stage_calls("assess_answer")) == 2
    assert reply.body.startswith(UNKNOWN_REPLY)
    assert reply.response_reason_code == "grounding_reject"


# ---- language ----------------------------------------------------------------------------


async def test_spanish_turn_is_answered_in_spanish_and_assessed(migrated_db) -> None:
    spanish = Draft("La mayoría de los resultados se informan en 24-48 horas.", EASY_BODY)
    client = FakeAnthropic(
        understand={
            "language": "es",
            "query": "How quickly are results available?",
        },
        draft=spanish,
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("¿Qué tan rápido llegan los resultados?")
        reply = await chat.last_bot()

    assert reply.body == spanish.text
    assert reply.source_chunk_ids
    # The search ran on the English query; the model was told the answer language.
    draft_call = client.stage_calls("draft")[0]
    assert '"language": "es"' in draft_call["messages"][-1]["content"][-1]["text"]
    # Translation is involved, so the assessor reviews it.
    assert len(client.stage_calls("assess_answer")) == 1


async def test_spanish_greeting_uses_the_models_reply(migrated_db) -> None:
    client = FakeAnthropic(
        understand={
            "language": "es",
            "intent": "smalltalk",
            "reply": "¡Hola! ¿En qué puedo ayudarte?",
        }
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("hola")
        reply = await chat.last_bot()

    assert reply.body == "¡Hola! ¿En qué puedo ayudarte?"
    assert client.stage_calls("draft") == []


# ---- canned replies ----------------------------------------------------------------------


async def _canned(session, site, shortcut="turnaround", body="Canned wording from staff."):
    from app.services.canned_bot import refresh_canned_embedding
    from tests.fakes import FakeEmbedder

    row = CannedReply(
        site_id=site.id,
        shortcut=shortcut,
        body=body,
        aliases=["how long until results"],
        enabled=True,
        bot_eligible=True,
    )
    await refresh_canned_embedding(row, FakeEmbedder())
    session.add(row)
    await session.commit()


async def test_canned_reply_is_skipped_for_a_visitor_in_another_language(migrated_db) -> None:
    client = FakeAnthropic(
        understand={"language": "es", "query": "how long until results"}, draft=GOOD_ANSWER
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await _canned(session, chat.site)
        await chat.say("¿Cuánto tardan los resultados?")
        reply = await chat.last_bot()

    assert reply.response_reason_code != "canned_reply"
    assert reply.body != "Canned wording from staff."
    assert len(client.stage_calls("draft")) == 1


async def test_rejected_canned_reply_falls_back_to_the_knowledge_path(migrated_db) -> None:
    client = FakeAnthropic(
        understand={"query": "how long until results"},
        draft=GOOD_ANSWER,
        assess=[assessment("reject", "irrelevant")],
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[("Unrelated page", "Something else.")])
        await _canned(session, chat.site)
        await chat.say("how long until results")
        reply = await chat.last_bot()

    # No relevant evidence for the draft, so the honest unknown reply follows the rejection;
    # it is never the generic "I couldn't verify" and never the rejected canned text.
    assert reply.body != "Canned wording from staff."
    assert client.stage_calls("assess_answer")


# ---- people ------------------------------------------------------------------------------


async def test_handoff_topic_match_opens_a_handoff_without_drafting(migrated_db) -> None:
    client = FakeAnthropic(
        understand={
            "matches_handoff_topic": True,
            "reply": "That needs a colleague. I'm bringing someone in.",
        }
    )
    async with session_maker()() as session:
        chat = await open_chat(
            session, client, chunks=[TIMING_CHUNK], handoff_topics=["refund disputes"]
        )
        await chat.say("I want my money back for the last order")
        reply = await chat.last_bot()
        conversation = await session.get(Conversation, chat.conversation.id)
        await session.refresh(conversation)

    assert reply.body == "That needs a colleague. I'm bringing someone in."
    assert conversation.state == "queued"
    assert conversation.escalation_reason == "policy_boundary"
    assert client.stage_calls("draft") == []
    assert understood(client)["site"]["handoff_topics"] == ["refund disputes"]


async def test_explicit_request_for_a_person_queues_the_chat(migrated_db) -> None:
    client = FakeAnthropic(understand={"intent": "handoff_request", "reply": "Of course."})
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("let me talk to a human")
        await session.refresh(chat.conversation)

    assert chat.conversation.state == "queued"
    assert chat.conversation.escalation_reason == "visitor_request"
    assert chat.conversation.intent == "handoff_request"


async def test_handoff_request_without_a_live_person_becomes_a_callback_with_the_contact_route(
    migrated_db,
) -> None:
    client = FakeAnthropic(
        understand={
            "intent": "handoff_request",
            "reply": "The team will follow up with you directly.",
        }
    )
    async with session_maker()() as session:
        chat = await open_chat(
            session, client, human_enabled=False, contact_info=["help@example.test"]
        )
        await chat.say("get me a human")
        reply = await chat.last_bot()
        await session.refresh(chat.conversation)
        lines = [row.body for row in await chat.rows("system")]

    assert reply.body == "The team will follow up with you directly.\n\nhelp@example.test"
    assert chat.conversation.state == "queued"
    assert chat.conversation.attention_needed is True
    assert CALLBACK_LINE in lines
    assert understood(client)["site"]["human_enabled"] is False


async def test_personal_case_goes_to_a_person_when_available(migrated_db) -> None:
    client = FakeAnthropic(
        understand={
            "intent": "personal_case",
            "own_record_quote": "my result",
            "reply": "A colleague will look.",
        }
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("Why was my application rejected?")
        await session.refresh(chat.conversation)

    assert chat.conversation.state == "queued"
    assert chat.conversation.escalation_reason == "individual_case"


async def test_accepting_an_offered_person_is_understood_from_the_previous_reply(
    migrated_db,
) -> None:
    client = FakeAnthropic(
        understand=[
            {},
            {"intent": "handoff_request", "reply": "Connecting you now."},
        ]
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("something nobody documented")
        await chat.say("yes please")
        await session.refresh(chat.conversation)

    second = json.loads(client.stage_calls("understand_turn")[1]["messages"][0]["content"])
    assert second["previous_reply_kind"] == "no_evidence"
    assert second["history"][-1]["role"] == "assistant"
    assert chat.conversation.state == "queued"


# ---- conversational turns ----------------------------------------------------------------


async def test_off_topic_reply_is_the_models_and_the_scope_was_supplied(migrated_db) -> None:
    client = FakeAnthropic(
        understand={
            "intent": "off_topic",
            "reply": "That's outside what I do. I can help with lab testing questions.",
        }
    )
    async with session_maker()() as session:
        chat = await open_chat(
            session, client, assistant_scope="lab testing questions", chunks=[TIMING_CHUNK]
        )
        await chat.say("write me a poem")
        reply = await chat.last_bot()

    assert reply.body.startswith("That's outside what I do.")
    assert reply.response_reason_code == "off_topic"
    assert understood(client)["site"]["assistant_scope"] == "lab testing questions"
    assert understood(client)["site"]["knowledge_topics"] == []
    assert client.stage_calls("draft") == []


async def test_without_a_configured_scope_the_model_is_given_knowledge_topics(
    migrated_db,
) -> None:
    client = FakeAnthropic()
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("hello")

    payload = understood(client)
    assert payload["site"]["assistant_scope"] == ""
    assert payload["site"]["knowledge_topics"] == [EASY_TITLE]


async def test_abuse_gets_a_boundary_reply_and_no_lookup(migrated_db) -> None:
    client = FakeAnthropic(understand={"intent": "abuse", "reply": "Let's keep this respectful."})
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("you are useless")
        reply = await chat.last_bot()

    assert reply.body == "Let's keep this respectful."
    assert reply.response_reason_code == "abuse"
    assert client.stage_calls("draft") == []


async def test_reply_with_invented_facts_is_not_shown(migrated_db) -> None:
    # A conversational reply may carry no number or link the visitor did not write.
    client = FakeAnthropic(
        understand={"intent": "smalltalk", "reply": "Hi! Results take 3 days: https://x.test"},
        draft=NO_ANSWER_DRAFT,
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("hi")
        reply = await chat.last_bot()

    assert "3 days" not in reply.body
    assert "x.test" not in reply.body


async def test_contact_request_shows_only_the_configured_details(migrated_db) -> None:
    client = FakeAnthropic(
        understand={"intent": "contact_request", "reply": "Here is how to reach us."}
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, contact_info=["help@example.test"])
        await chat.say("how do I contact you?")
        reply = await chat.last_bot()

    assert reply.body == "Here is how to reach us.\n\nhelp@example.test"
    assert client.stage_calls("draft") == []


async def test_contact_request_without_configured_details_uses_the_knowledge(
    migrated_db,
) -> None:
    client = FakeAnthropic(understand={"intent": "contact_request"}, draft=NO_ANSWER_DRAFT)
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[("Contact", "Email help@example.test.")])
        await chat.say("how do I contact you?")

    assert len(client.stage_calls("draft")) == 1


async def test_ambiguous_question_asks_one_clarifying_question_without_searching(
    migrated_db,
) -> None:
    client = FakeAnthropic(
        understand={
            "ambiguity": "an employer account or a candidate",
            "reply": "Which one do you mean?",
        }
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("how do I get started?")
        reply = await chat.last_bot()

    assert reply.body == "Which one do you mean?"
    assert reply.response_outcome == "clarification"
    assert client.stage_calls("draft") == []


async def test_identifier_in_the_message_is_masked_and_the_model_asks_not_to_share(
    migrated_db,
) -> None:
    client = FakeAnthropic(
        understand={
            "intent": "shared_identifiers",
            "reply": "Please don't share personal identifiers here.",
        }
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("My number is 123-45-6789, can you look me up?")
        reply = await chat.last_bot()

    payload = understood(client)
    assert payload["identifiers_redacted"] is True
    assert "123-45-6789" not in json.dumps(payload)
    assert reply.body == "Please don't share personal identifiers here."
    assert reply.system_reason == "sensitive"


async def test_source_request_shows_the_previous_answers_source(migrated_db) -> None:
    client = FakeAnthropic(
        understand=[{}, {"intent": "source_request", "reply": "Here is the source."}],
        draft=GOOD_ANSWER,
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")
        await chat.say("where did you get that?")
        reply = await chat.last_bot()

    assert reply.body.startswith("Here is the source.\n\nhttps://")
    assert EASY_BODY in reply.body
    assert reply.source_chunk_ids
    assert len(client.stage_calls("draft")) == 1


async def test_staff_turns_are_marked_in_the_history_the_model_sees(migrated_db) -> None:
    from uuid import uuid4

    from app.models.message import Message
    from app.models.user import User

    client = FakeAnthropic(understand={"intent": "smalltalk", "reply": "Hello again."})
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        staff = User(email="staff@local.test", display_name="Sam", password_hash="unused")
        session.add(staff)
        await session.flush()
        session.add(
            Message(
                conversation_id=chat.conversation.id,
                site_id=chat.site.id,
                role="agent",
                body="We will refund that.",
                client_message_id=uuid4(),
                author_user_id=staff.id,
            )
        )
        await session.commit()
        await chat.say("thanks")

    history = understood(client)["history"]
    assert history[0] == {"role": "assistant", "content": "[Staff member] We will refund that."}


# ---- providers fail honestly -------------------------------------------------------------


async def test_understanding_failure_is_an_honest_technical_failure(migrated_db) -> None:
    client = FakeAnthropic(fail={"understand_turn"}, draft=GOOD_ANSWER)
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()

    assert reply.body == TECH_FAIL_HUMAN
    assert reply.response_reason_code == "tech_fail"
    assert client.stage_calls("draft") == []


async def test_draft_failure_is_an_honest_technical_failure_without_a_person(
    migrated_db,
) -> None:
    client = FakeAnthropic(fail={"draft"})
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK], human_enabled=False)
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()

    assert reply.body == TECH_FAIL_SOLO
    assert reply.source_chunk_ids is None


async def test_invalid_understanding_output_is_a_technical_failure(migrated_db) -> None:
    client = FakeAnthropic(understand={"intent": "not_an_intent"})
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()

    assert reply.body == TECH_FAIL_HUMAN


async def test_assessor_outage_does_not_block_a_plain_continuation_free_answer(
    migrated_db,
) -> None:
    # A continuation needs the assessor; when it is down the bot says so rather than guess.
    client = FakeAnthropic(
        draft=GOOD_ANSWER, understand={"relation": "continuation"}, fail={"assess_answer"}
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()

    assert reply.body == TECH_FAIL_HUMAN


async def test_knowledge_never_crosses_sites(migrated_db) -> None:
    client = FakeAnthropic(draft=GOOD_ANSWER)
    async with session_maker()() as session:
        other = await open_chat(session, client, key="other-site", chunks=[TIMING_CHUNK])
        chat = await open_chat(session, client, key="empty-site")
        await chat.say("How quickly are results available?")
        reply = await chat.last_bot()
        del other

    assert client.stage_calls("draft") == []
    assert reply.body.startswith(UNKNOWN_REPLY)


# ---- state across turns --------------------------------------------------------------------


async def test_a_cited_answer_resets_the_miss_count(migrated_db) -> None:
    client = FakeAnthropic(draft=GOOD_ANSWER)
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("something nobody documented")
        assert chat.conversation.fallback_count == 1
        await chat.say("How quickly are results available?")

    assert chat.conversation.fallback_count == 0


async def test_consecutive_misses_are_marked_repeated_and_still_offer_a_person(
    migrated_db,
) -> None:
    client = FakeAnthropic()
    async with session_maker()() as session:
        chat = await open_chat(session, client)
        await chat.say("first unknown thing")
        await chat.say("second unknown thing")
        rows = await chat.rows("bot")

    assert [row.response_reason_code for row in rows] == ["no_evidence", "repeated_miss"]
    assert chat.conversation.state == "bot"


async def test_agreement_to_something_other_than_a_person_does_not_queue(migrated_db) -> None:
    client = FakeAnthropic(
        understand=[
            {"ambiguity": "an account or a test", "reply": "Which do you mean?"},
            {"relation": "continuation", "query": "account setup"},
        ],
        draft=NO_ANSWER_DRAFT,
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("how do I start?")
        await chat.say("yes")

    second = json.loads(client.stage_calls("understand_turn")[1]["messages"][0]["content"])
    assert second["previous_reply_was_clarification"] is True
    assert chat.conversation.state == "bot"


async def test_overview_intent_asks_the_retriever_for_an_overview(migrated_db, monkeypatch) -> None:
    from app.services.kb_hybrid import HybridKbSearch

    seen: list[bool] = []
    original = HybridKbSearch.search_with_deferred_embed

    async def spy(self, *args, **kwargs):
        seen.append(kwargs.get("overview"))
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(HybridKbSearch, "search_with_deferred_embed", spy)
    client = FakeAnthropic(understand={"intent": "overview"}, draft=NO_ANSWER_DRAFT)
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("what do you do?")

    assert seen == [True]


async def test_a_turn_searches_the_knowledge_once(migrated_db, monkeypatch) -> None:
    from app.services.kb_hybrid import HybridKbSearch

    searches: list[str] = []
    original = HybridKbSearch.search_with_deferred_embed

    async def spy(self, embedder, site_id, query, *args, **kwargs):
        searches.append(query)
        return await original(self, embedder, site_id, query, *args, **kwargs)

    monkeypatch.setattr(HybridKbSearch, "search_with_deferred_embed", spy)
    client = FakeAnthropic(draft=GOOD_ANSWER, understand={"query": "turnaround of results"})
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await _canned(session, chat.site)
        await chat.say("How quickly are results available?")

    assert searches == ["turnaround of results"]


async def test_continuation_keeps_the_previous_answers_evidence(migrated_db) -> None:
    client = FakeAnthropic(
        understand=[{}, {"relation": "continuation", "query": "and then?"}],
        draft=[GOOD_ANSWER, GOOD_ANSWER],
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[TIMING_CHUNK])
        await chat.say("How quickly are results available?")
        first = await chat.last_bot()
        await chat.say("and after that?")
        second = await chat.last_bot()

    # The follow-up's own search finds nothing relevant; the previous passage is carried.
    assert second.source_chunk_ids == first.source_chunk_ids
    assert len(client.stage_calls("draft")) == 2
