"""Retrieval, canned replies and bot turns never cross sites.

Two sites publish the same question and answer verbatim (and use the same canned shortcut), so
a retrieval path that forgets its site filter returns the other site's row and fails here.
"""

import uuid

from sqlalchemy import select

from app.db import session_maker
from app.models.canned_reply import CannedReply
from app.models.message import Message
from app.models.message_citation import MessageCitation
from app.services.canned_bot import canned_reply_decision, embed_replies, search_canned
from app.services.conversation_service import ConversationService
from app.services.kb_hybrid import HybridKbSearch
from tests.site_isolation_helpers import (
    ALPHA_ONLY_QUESTION,
    BETA_ONLY_ANSWER,
    BETA_ONLY_QUESTION,
    SHARED_ANSWER,
    SHARED_QUESTION,
    ForeignCitingResponder,
    TopicEmbedder,
    add_bot_conversation,
    add_live_chunk,
    add_site,
    seed_two_sites,
)
from tests.ws_helpers import HOST_ORIGIN

EMBEDDER = TopicEmbedder()
ALPHA_HOURS = "Alpha is open 9 to 5 on weekdays."
BETA_HOURS = "Beta is open 8 to 6 every day."
ALPHA_PRIVATE = "Alpha-only canned: invoices are issued on the first of the month."
GENERAL_BODY = "General script: please contact support for anything else."


async def _turn(session, site, text: str, responder: ForeignCitingResponder):
    visitor, conversation = await add_bot_conversation(session, site)
    await session.commit()
    service = ConversationService(session, responder=responder, embedder=EMBEDDER)
    result = await service.visitor_message(
        conversation.id, visitor.id, HOST_ORIGIN, uuid.uuid4(), text
    )
    assert result.generation_id is not None
    await service.run_bot_turn(conversation.id, result.generation_id)
    bot_rows = list(
        (
            await session.scalars(
                select(Message)
                .where(
                    Message.conversation_id == conversation.id,
                    Message.role.in_(("bot", "system")),
                )
                .order_by(Message.id)
            )
        ).all()
    )
    citations = list(
        (
            await session.scalars(
                select(MessageCitation).where(
                    MessageCitation.message_id.in_([row.id for row in bot_rows])
                )
            )
        ).all()
    )
    return bot_rows, citations


# --- knowledge retrieval -------------------------------------------------------------------


async def test_identical_faq_on_two_sites_is_retrieved_per_site(migrated_db) -> None:
    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        vector = EMBEDDER.vector(SHARED_QUESTION)
        search = HybridKbSearch(session)
        for site, own, other in (
            (seeded.alpha, seeded.alpha_shared, seeded.beta_shared),
            (seeded.beta, seeded.beta_shared, seeded.alpha_shared),
        ):
            hits = await search.search(site.id, SHARED_QUESTION, vector)
            assert [hit.id for hit in hits] == [own.id]
            assert all(hit.site_id == site.id for hit in hits)
            assert other.id not in {hit.id for hit in hits}


async def test_every_retrieval_arm_stays_inside_the_requesting_site(migrated_db) -> None:
    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        search = HybridKbSearch(session)
        # Beta's private FAQ asked on Alpha: lexical arm, dense arm and exact match.
        vector = EMBEDDER.vector(BETA_ONLY_QUESTION)
        beta_ids = {seeded.beta_only.id, seeded.beta_shared.id}
        for kwargs in ({"query_vector": vector}, {"query_vector": None}):
            hits = await search.search(seeded.alpha.id, BETA_ONLY_QUESTION, **kwargs)
            assert not beta_ids & {hit.id for hit in hits}
        deferred = await search.search_with_deferred_embed(
            EMBEDDER, seeded.alpha.id, BETA_ONLY_QUESTION
        )
        assert not beta_ids & {hit.id for hit in deferred}
        exact = await search._exact_match(seeded.alpha.id, BETA_ONLY_QUESTION)
        assert exact is None
        # And the owner still finds it, so the empty result above is the filter, not a miss.
        own = await search.search_with_deferred_embed(EMBEDDER, seeded.beta.id, BETA_ONLY_QUESTION)
        assert seeded.beta_only.id in {hit.id for hit in own}


async def test_disabling_a_chunk_does_not_hide_the_same_text_on_another_site(migrated_db) -> None:
    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        chunk = await session.get(type(seeded.alpha_shared), seeded.alpha_shared.id)
        chunk.enabled = False
        await session.commit()
        search = HybridKbSearch(session)
        vector = EMBEDDER.vector(SHARED_QUESTION)
        assert await search.search(seeded.alpha.id, SHARED_QUESTION, vector) == []
        beta_hits = await search.search(seeded.beta.id, SHARED_QUESTION, vector)
        assert [hit.id for hit in beta_hits] == [seeded.beta_shared.id]


# --- bot turns -----------------------------------------------------------------------------


async def test_bot_turn_offers_and_cites_only_its_own_sites_chunks(migrated_db) -> None:
    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        for site, own, foreign in (
            (seeded.alpha, {seeded.alpha_shared.id, seeded.alpha_only.id}, seeded.beta),
            (seeded.beta, {seeded.beta_shared.id, seeded.beta_only.id}, seeded.alpha),
        ):
            responder = ForeignCitingResponder()
            rows, citations = await _turn(session, site, SHARED_QUESTION, responder)
            assert responder.calls, "the model should have been asked"
            assert responder.offered_ids <= own
            assert foreign.name.split()[0] not in responder.offered_text
            cited = {chunk_id for row in rows for chunk_id in (row.source_chunk_ids or [])}
            cited |= {citation.chunk_id for citation in citations}
            assert cited and cited <= own


async def test_bot_never_answers_from_another_sites_private_faq(migrated_db) -> None:
    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        responder = ForeignCitingResponder()
        rows, citations = await _turn(session, seeded.alpha, BETA_ONLY_QUESTION, responder)
        beta_ids = {seeded.beta_only.id, seeded.beta_shared.id}
        assert not beta_ids & responder.offered_ids
        assert BETA_ONLY_ANSWER not in responder.offered_text
        assert all(BETA_ONLY_ANSWER not in row.body for row in rows)
        assert not beta_ids & {
            chunk_id for row in rows for chunk_id in (row.source_chunk_ids or [])
        }
        assert not beta_ids & {citation.chunk_id for citation in citations}


async def test_citation_to_a_chunk_of_another_site_is_never_persisted(migrated_db) -> None:
    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        responder = ForeignCitingResponder(
            cite_chunk_id=seeded.beta_only.id, cite_snapshot_id=seeded.beta_only.snapshot_id
        )
        rows, citations = await _turn(session, seeded.alpha, ALPHA_ONLY_QUESTION, responder)
        assert seeded.beta_only.id not in {c.chunk_id for c in citations}
        assert seeded.beta_only.id not in {
            chunk_id for row in rows for chunk_id in (row.source_chunk_ids or [])
        }
        assert all("foreign evidence" not in row.body for row in rows)


# --- canned replies ------------------------------------------------------------------------


async def _seed_canned(session):
    alpha = await add_site(session, "alpha", "Alpha Co")
    beta = await add_site(session, "beta", "Beta Co")
    shared = {"shortcut": "business_hours", "aliases": ["opening_hours"]}
    alpha_hours = CannedReply(site_id=alpha.id, body=ALPHA_HOURS, **shared)
    beta_hours = CannedReply(site_id=beta.id, body=BETA_HOURS, **shared)
    alpha_private = CannedReply(site_id=alpha.id, shortcut="invoice_dates", body=ALPHA_PRIVATE)
    general = CannedReply(site_id=None, shortcut="anything_else", body=GENERAL_BODY)
    session.add_all([alpha_hours, beta_hours, alpha_private, general])
    await session.flush()
    await embed_replies([alpha_hours, beta_hours, alpha_private, general], EMBEDDER)
    await session.commit()
    return alpha, beta, alpha_hours, beta_hours, alpha_private, general


async def test_canned_search_returns_only_the_requesting_sites_reply(migrated_db) -> None:
    async with session_maker()() as session:
        alpha, beta, alpha_hours, beta_hours, alpha_private, _general = await _seed_canned(session)
        for site, own, other in ((alpha, alpha_hours, beta_hours), (beta, beta_hours, alpha_hours)):
            hits = await search_canned(session, EMBEDDER, site.id, "What are your business hours?")
            assert own.id in {hit.id for hit in hits}
            assert other.id not in {hit.id for hit in hits}
        beta_hits = await search_canned(session, EMBEDDER, beta.id, "When are invoice dates?")
        assert alpha_private.id not in {hit.id for hit in beta_hits}


async def test_canned_decision_uses_own_body_for_a_shared_shortcut(migrated_db) -> None:
    async with session_maker()() as session:
        alpha, beta, *_ = await _seed_canned(session)
        for site, expected, forbidden in (
            (alpha, ALPHA_HOURS, BETA_HOURS),
            (beta, BETA_HOURS, ALPHA_HOURS),
        ):
            decision = await canned_reply_decision(
                session, EMBEDDER, site.id, "business hours", None
            )
            assert decision is not None
            assert decision.body == expected
            assert forbidden not in decision.body


async def test_canned_bot_turn_replies_with_the_sites_own_script(migrated_db) -> None:
    async with session_maker()() as session:
        alpha, beta, *_ = await _seed_canned(session)
        for site, expected, forbidden in (
            (alpha, ALPHA_HOURS, BETA_HOURS),
            (beta, BETA_HOURS, ALPHA_HOURS),
        ):
            responder = ForeignCitingResponder()
            rows, _ = await _turn(session, site, "business hours", responder)
            assert [row.body for row in rows] == [expected]
            assert forbidden not in rows[0].body


async def test_no_site_bot_uses_general_replies_and_site_replies_stay_private(
    migrated_db,
) -> None:
    """General (no site) scripts are staff shortcuts only; the bot sends a site's own wording."""
    async with session_maker()() as session:
        alpha, beta, _ah, _bh, alpha_private, general = await _seed_canned(session)
        rows = {site.key: await _bot_eligible(session, site.id) for site in (alpha, beta)}
        assert general.id not in rows["alpha"] and general.id not in rows["beta"]
        assert alpha_private.id in rows["alpha"]
        assert alpha_private.id not in rows["beta"]
        for site in (alpha, beta):
            hits = await search_canned(session, EMBEDDER, site.id, GENERAL_BODY)
            assert general.id not in {hit.id for hit in hits}


async def _bot_eligible(session, site_id) -> set[uuid.UUID]:
    from app.repositories.canned_reply_repo import CannedReplyRepository

    return {row.id for row in await CannedReplyRepository(session).list_bot_eligible(site_id)}


async def test_site_with_no_knowledge_does_not_borrow_another_sites(migrated_db) -> None:
    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        empty = await add_site(session, "empty", "Empty Co")
        await add_live_chunk(session, empty, "Unrelated", "Unrelated.")
        await session.commit()
        responder = ForeignCitingResponder()
        rows, citations = await _turn(session, empty, SHARED_QUESTION, responder)
        own_ids = {seeded.alpha_shared.id, seeded.beta_shared.id}
        assert not own_ids & responder.offered_ids
        assert SHARED_ANSWER not in responder.offered_text
        assert all(SHARED_ANSWER not in row.body for row in rows)
        assert not own_ids & {citation.chunk_id for citation in citations}
