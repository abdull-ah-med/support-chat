"""Canned replies are approved bot wording, not a rewrite of the knowledge base.

Oracles are the LiveChat SampleSite library: turnaround_results body, the
enrollment miss (no enrollment canned), and greeting rows that must stay staff-only.
"""

# LiveChat export copy uses en dashes and curly quotes. Those characters are the
# approved body, not a typing substitute.
# ruff: noqa: RUF001

import uuid

from sqlalchemy import select

from app.db import session_maker
from app.models.canned_reply import CannedReply
from app.models.message import Message
from app.services.conversation_service import ConversationService
from tests.bot_fixtures import (
    ADA_NAME,
    RecordingGroundedResponder,
    insert_bot_conversation,
    insert_chunk,
    insert_site,
)
from tests.fakes import FakeEmbedder
from tests.ws_helpers import HOST_ORIGIN

TURNAROUND_BODY = (
    "Most negative results are reported within 24–48 hours, and rapid testing can "
    "deliver negatives in minutes.\n"
    "Non-negatives always go to a certified lab with MRO review before being reported."
)
WEBSITE_REWRITE = "The website says results often come back in a few days."
ENROLLMENT_PAGE = (
    "The enrollment process is three steps with no paper involved. Enter the "
    "candidate's name and email in the portal and send it to us."
)
NEXT_STEP_BODY = (
    "Perfect — the quickest next step is to submit a short inquiry form. After you "
    "submit, you’ll receive an email with our Calendly link to schedule a call.\n"
    "Here’s the form: https://miurl.cc/eyVVpl5mI"
)
HELLO_BODY = (
    "Hello! I hope you are doing well today. My name is Jo, I am a LIVE agent "
    "here to assist you. How may I help you today?"
)
NAMED_BODY = "Hi %customer-name%, most negative results are reported within 24–48 hours."
DOT_AGENCY_BODY = (
    "Thanks — that helps. Since you’re DOT-regulated under {DOTAgency}, we’ll get you "
    "pointed to the correct program."
)
LLM_TRAP = "THIS TEXT MEANS THE MODEL REWROTE THE ANSWER"
TURNAROUND_NEXTSTEP_BODY = (
    TURNAROUND_BODY + "\nTo confirm the best option for your program, submit the inquiry form:\n"
    "https://miurl.cc/eyVVpl5mI"
)
DER_WHAT_IS_BODY = (
    "A DER (Designated Employer Representative) is the person who manages the DOT "
    "testing program on the employer side (coordination, notifications, and compliance "
    "follow-through).\n"
    "Do you already have a DER? Yes / No / Not sure"
)
DER_FOLLOWUP_BODY = (
    "Perfect — having a DER simplifies setup. We’ll coordinate notifications and "
    "responsibilities around your existing DER process."
)
PRICING_SHORT_BODY = (
    "Pricing depends on DOT status, headcount/volume, and selected services/add-ons."
)
DOT_PROGRAM_BODY = (
    "Yes. We support employers with DOT drug and alcohol testing, random pool "
    "management, MIS reporting, and DOT physicals."
)


async def _insert_canned(
    session,
    *,
    site_id,
    shortcut: str,
    body: str,
    enabled: bool = True,
    bot_eligible: bool = True,
    aliases: list[str] | None = None,
) -> CannedReply:
    reply = CannedReply(
        site_id=site_id,
        shortcut=shortcut,
        body=body,
        enabled=enabled,
        aliases=list(aliases or []),
        bot_eligible=bot_eligible,
    )
    session.add(reply)
    await session.flush()
    return reply


async def _bot_reply(session, conversation_id) -> Message:
    messages = list(
        (
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id, Message.role == "bot")
                .order_by(Message.id)
            )
        ).all()
    )
    assert messages
    return messages[-1]


async def test_matching_turnaround_canned_is_sent_verbatim_without_the_model(migrated_db) -> None:
    """Catches the bot paraphrasing an approved turnaround script from the website."""
    async with session_maker()() as session:
        site = await insert_site(session, "samplesite", "SampleSite")
        chunk = await insert_chunk(
            session,
            site,
            "How quickly are results available?",
            "Most negative results are reported within 24-48 hours.",
        )
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="turnaround_results",
            body=TURNAROUND_BODY,
            aliases=["how long until drug test results come back"],
        )
        visitor, conversation = await insert_bot_conversation(session, site)
        await session.commit()
        responder = RecordingGroundedResponder(
            body=WEBSITE_REWRITE,
            evidence_id=chunk.id,
            snapshot_id=chunk.snapshot_id,
            source_title="How quickly are results available?",
            source_url="https://legacy.test/turnaround",
            cited_text="Most negative results are reported within 24-48 hours.",
        )
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "How long until drug test results come back?",
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation.id, result.generation_id)
        reply = await _bot_reply(session, conversation.id)

    assert reply.body == TURNAROUND_BODY
    assert reply.system_reason == "canned"
    assert reply.display_locator == "#turnaround_results"
    assert reply.source_urls is None
    assert reply.source_chunk_ids is None
    assert responder.draft_calls() == []


async def test_enrollment_question_does_not_send_a_routing_form_canned(migrated_db) -> None:
    """Catches a form-link canned stealing the enrollment miss the meeting called out."""
    async with session_maker()() as session:
        site = await insert_site(session, "samplesite", "SampleSite")
        chunk = await insert_chunk(
            session,
            site,
            "Recruiter steps after enrollment",
            ENROLLMENT_PAGE,
        )
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="next_step_universal",
            body=NEXT_STEP_BODY,
        )
        visitor, conversation = await insert_bot_conversation(session, site)
        await session.commit()
        responder = RecordingGroundedResponder(
            body=LLM_TRAP,
            evidence_id=chunk.id,
            snapshot_id=chunk.snapshot_id,
            source_title="Recruiter steps after enrollment",
            source_url="https://legacy.test/enrollment",
            cited_text=ENROLLMENT_PAGE,
        )
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "What is the enrollment process?",
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation.id, result.generation_id)
        reply = await _bot_reply(session, conversation.id)

    assert "miurl.cc/eyVVpl5mI" not in reply.body
    assert reply.body != NEXT_STEP_BODY
    assert reply.system_reason != "canned"


async def test_greeting_and_other_site_canned_stay_out_of_bot_replies(migrated_db) -> None:
    """Catches idle/hello scripts or Verification Checks copy leaking into SampleSite."""
    async with session_maker()() as session:
        easy = await insert_site(session, "samplesite", "SampleSite")
        background = await insert_site(session, "verificationchecks", "Verification Checks")
        chunk = await insert_chunk(
            session,
            easy,
            "How quickly are results available?",
            "Most negative results are reported within 24-48 hours.",
        )
        await _insert_canned(
            session,
            site_id=None,
            shortcut="hello",
            body=HELLO_BODY,
            bot_eligible=False,
        )
        await _insert_canned(
            session,
            site_id=background.id,
            shortcut="turnaround_results",
            body=TURNAROUND_BODY,
        )
        visitor, conversation = await insert_bot_conversation(session, easy)
        await session.commit()
        responder = RecordingGroundedResponder(
            body=WEBSITE_REWRITE,
            evidence_id=chunk.id,
            snapshot_id=chunk.snapshot_id,
            source_title="How quickly are results available?",
            source_url="https://legacy.test/turnaround",
            cited_text="Most negative results are reported within 24-48 hours.",
        )
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        hello = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "hello",
        )
        if hello.generation_id is not None:
            await service.run_bot_turn(conversation.id, hello.generation_id)
        follow = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "How long until drug test results come back?",
        )
        assert follow.generation_id is not None
        await service.run_bot_turn(conversation.id, follow.generation_id)
        bodies = [
            row.body
            for row in (
                await session.scalars(
                    select(Message).where(
                        Message.conversation_id == conversation.id,
                        Message.role.in_(("bot", "system")),
                    )
                )
            ).all()
        ]
        last = await _bot_reply(session, conversation.id)

    assert HELLO_BODY not in bodies
    assert last.body != TURNAROUND_BODY
    assert last.system_reason != "canned"


async def test_canned_fills_visitor_name_and_strips_unknown_placeholders(migrated_db) -> None:
    """Catches the bot inventing an agent name or leaving LiveChat placeholders in the reply."""
    async with session_maker()() as session:
        site = await insert_site(session, "samplesite", "SampleSite")
        await insert_chunk(
            session,
            site,
            "How quickly are results available?",
            "Most negative results are reported within 24-48 hours.",
        )
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="turnaround_results",
            body=NAMED_BODY + " Ask %agent-name% if needed.",
            aliases=["how long until results come back"],
        )
        visitor, conversation = await insert_bot_conversation(session, site)
        await session.commit()
        responder = RecordingGroundedResponder(
            body=LLM_TRAP,
            evidence_id=uuid.uuid4(),
            snapshot_id=None,
            source_title="unused",
            source_url="https://legacy.test/unused",
            cited_text="unused",
        )
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "How long until results come back?",
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation.id, result.generation_id)
        reply = await _bot_reply(session, conversation.id)

    assert reply.body == (
        f"Hi {ADA_NAME}, most negative results are reported within 24–48 hours. Ask if needed."
    )
    assert "%customer-name%" not in reply.body
    assert "%agent-name%" not in reply.body
    assert responder.draft_calls() == []


async def test_turnaround_faq_beats_the_followup_form_script(migrated_db) -> None:
    """Catches the next-step form canned winning over the actual turnaround answer."""
    async with session_maker()() as session:
        site = await insert_site(session, "samplesite", "SampleSite")
        await insert_chunk(
            session,
            site,
            "How quickly are results available?",
            "Most negative results are reported within 24-48 hours.",
        )
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="turnaround_results_nextstep",
            body=TURNAROUND_NEXTSTEP_BODY,
            aliases=["next step", "submit the inquiry form"],
        )
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="turnaround_results",
            body=TURNAROUND_BODY,
            aliases=["how long until drug test results come back"],
        )
        visitor, conversation = await insert_bot_conversation(session, site)
        await session.commit()
        responder = RecordingGroundedResponder(
            body=WEBSITE_REWRITE,
            evidence_id=uuid.uuid4(),
            snapshot_id=None,
            source_title="unused",
            source_url="https://legacy.test/unused",
            cited_text="unused",
        )
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "How long until drug test results come back?",
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation.id, result.generation_id)
        reply = await _bot_reply(session, conversation.id)

    assert reply.body == TURNAROUND_BODY
    assert "miurl.cc" not in reply.body
    assert reply.display_locator == "#turnaround_results"


async def test_der_definition_beats_the_existing_der_followup_script(migrated_db) -> None:
    """Catches 'what is a DER' sending the already-have-a-DER routing script."""
    async with session_maker()() as session:
        site = await insert_site(session, "samplesite", "SampleSite")
        await insert_chunk(
            session,
            site,
            "What is a DER?",
            "A DER manages the employer side of a DOT testing program.",
        )
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="der_yes_followup",
            body=DER_FOLLOWUP_BODY,
            aliases=["yes"],
        )
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="der_what_is",
            body=DER_WHAT_IS_BODY,
            aliases=["what is a der"],
        )
        visitor, conversation = await insert_bot_conversation(session, site)
        await session.commit()
        responder = RecordingGroundedResponder(
            body=LLM_TRAP,
            evidence_id=uuid.uuid4(),
            snapshot_id=None,
            source_title="unused",
            source_url="https://legacy.test/unused",
            cited_text="unused",
        )
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "What is a DER?",
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation.id, result.generation_id)
        reply = await _bot_reply(session, conversation.id)

    assert reply.body == DER_WHAT_IS_BODY
    assert reply.display_locator == "#der_what_is"
    assert "simplifies setup" not in reply.body


async def test_canned_with_agent_fill_in_fields_is_never_sent_to_the_visitor(migrated_db) -> None:
    """Catches '{DOTAgency}' reaching a visitor because only an agent can fill that field."""
    async with session_maker()() as session:
        site = await insert_site(session, "samplesite", "SampleSite")
        chunk = await insert_chunk(
            session,
            site,
            "Which DOT agency applies?",
            "Most DOT-regulated trucking is FMCSA.",
        )
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="dot_yes_agency_known",
            body=DOT_AGENCY_BODY,
        )
        visitor, conversation = await insert_bot_conversation(session, site)
        await session.commit()
        responder = RecordingGroundedResponder(
            body=WEBSITE_REWRITE,
            evidence_id=chunk.id,
            snapshot_id=chunk.snapshot_id,
            source_title="Which DOT agency applies?",
            source_url="https://legacy.test/dot-agency",
            cited_text="Most DOT-regulated trucking is FMCSA.",
        )
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "Yes the DOT agency is known",
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation.id, result.generation_id)
        reply = await _bot_reply(session, conversation.id)

    assert "{DOTAgency}" not in reply.body
    assert reply.system_reason != "canned"


async def test_cost_question_sends_pricing_not_the_dot_program_script(migrated_db) -> None:
    """Catches 'How much does DOT testing cost?' being answered with the 49 CFR program spiel."""
    async with session_maker()() as session:
        site = await insert_site(session, "samplesite", "SampleSite")
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="pricing_short",
            body=PRICING_SHORT_BODY,
            aliases=["how much does testing cost"],
        )
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="dot_compliant_testing",
            body=DOT_PROGRAM_BODY,
            aliases=["dot compliant testing"],
        )
        visitor, conversation = await insert_bot_conversation(session, site)
        await session.commit()
        service = ConversationService(
            session,
            responder=RecordingGroundedResponder(
                body=LLM_TRAP,
                evidence_id=uuid.uuid4(),
                snapshot_id=None,
                source_title="unused",
                source_url="https://legacy.test/unused",
                cited_text="unused",
            ),
            embedder=FakeEmbedder(),
        )
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "How much does DOT testing cost?",
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation.id, result.generation_id)
        reply = await _bot_reply(session, conversation.id)

    assert reply.body == PRICING_SHORT_BODY
    assert reply.system_reason == "canned"
    assert reply.display_locator == "#pricing_short"


async def test_bare_cost_question_sends_the_pricing_canned(migrated_db) -> None:
    """Catches 'How much does it cost?' falling through because no row contained the word pricing."""
    async with session_maker()() as session:
        site = await insert_site(session, "samplesite", "SampleSite")
        await _insert_canned(
            session,
            site_id=site.id,
            shortcut="pricing_short",
            body=PRICING_SHORT_BODY,
            aliases=["how much does it cost"],
        )
        visitor, conversation = await insert_bot_conversation(session, site)
        await session.commit()
        service = ConversationService(
            session,
            responder=RecordingGroundedResponder(
                body=LLM_TRAP,
                evidence_id=uuid.uuid4(),
                snapshot_id=None,
                source_title="unused",
                source_url="https://legacy.test/unused",
                cited_text="unused",
            ),
            embedder=FakeEmbedder(),
        )
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            "How much does it cost?",
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation.id, result.generation_id)
        reply = await _bot_reply(session, conversation.id)

    assert reply.body == PRICING_SHORT_BODY
    assert reply.display_locator == "#pricing_short"
