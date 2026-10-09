"""Suggested FAQs: the same question asked on two sites is two gaps, handled independently."""

import uuid

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api.knowledge_gaps import get_embedder
from app.db import session_maker
from app.main import app
from app.models.canned_reply import CannedReply
from app.models.knowledge_gap import KnowledgeGap, KnowledgeGapHit
from app.services.knowledge_gap_service import KnowledgeGapService
from tests.knowledge_gap_fixtures import (
    agent_message,
    auth,
    db,
    seed_chunk,
    seed_conversation,
    seed_gap,
    seed_hit,
)
from tests.site_isolation_helpers import TopicEmbedder, add_bot_conversation, add_site
from tests.ws_helpers import (
    ALEX_EMAIL,
    ALEX_NAME,
    ALEX_PASSWORD,
    HOST_ORIGIN,
    insert_site,
    insert_staff,
    login_staff,
)

QUESTION = "Where can I download my invoice?"
ANSWER = "Download invoices from Billing."
EMBEDDER = TopicEmbedder()
ADMIN_EMAIL = "admin@example.local"


def _world(client: TestClient) -> tuple[uuid.UUID, uuid.UUID, str, str]:
    alpha = insert_site("alpha", "Alpha Co", "a" * 64, [HOST_ORIGIN])
    beta = insert_site("beta", "Beta Co", "b" * 64, [HOST_ORIGIN])
    insert_staff(ALEX_EMAIL, ALEX_NAME, ALEX_PASSWORD)
    insert_staff(ADMIN_EMAIL, "Admin", ALEX_PASSWORD, is_admin=True)
    return alpha, beta, login_staff(client), login_staff(client, ADMIN_EMAIL, ALEX_PASSWORD)


async def _hit(session, site, question: str):
    from app.models.message import Message
    from app.repositories.knowledge_gap_repo import KnowledgeGapRepository

    _visitor, conversation = await add_bot_conversation(session, site)
    session.add(
        Message(
            conversation_id=conversation.id,
            site_id=site.id,
            role="visitor",
            client_message_id=uuid.uuid4(),
            body=question,
        )
    )
    reply = Message(
        conversation_id=conversation.id,
        site_id=site.id,
        role="bot",
        body="Could you rephrase that?",
        system_reason="clarify",
    )
    session.add(reply)
    await session.flush()
    await KnowledgeGapRepository(session).add_hit(
        site_id=site.id,
        conversation_id=conversation.id,
        message_id=reply.id,
        question=question,
        reason="no_evidence",
    )
    await session.commit()


async def test_clustering_never_merges_identical_questions_across_sites(migrated_db) -> None:
    """Same words and the same vector: the only thing separating the gaps is the site."""
    async with session_maker()() as session:
        alpha = await add_site(session, "alpha", "Alpha Co")
        beta = await add_site(session, "beta", "Beta Co")
        await session.commit()
        for site in (alpha, beta, alpha, beta, alpha):
            await _hit(session, site, QUESTION)
        service = KnowledgeGapService(session)
        while await service.assign_next(EMBEDDER):
            pass
        gaps = list((await session.scalars(select(KnowledgeGap))).all())
        assert sorted(gap.site_id for gap in gaps) == sorted([alpha.id, beta.id])
        hits = list((await session.scalars(select(KnowledgeGapHit))).all())
        gap_site = {gap.id: gap.site_id for gap in gaps}
        assert len(hits) == 5
        assert all(hit.gap_id is not None for hit in hits)
        assert all(gap_site[hit.gap_id] == hit.site_id for hit in hits)


async def test_clustering_by_meaning_stays_on_the_site(migrated_db) -> None:
    """A paraphrase (same vector, different words) joins its own site's gap, not the other's."""
    async with session_maker()() as session:
        alpha = await add_site(session, "alpha", "Alpha Co")
        beta = await add_site(session, "beta", "Beta Co")
        await session.commit()
        await _hit(session, beta, "Need an invoice copy")
        service = KnowledgeGapService(session)
        await service.assign_next(EMBEDDER)
        await _hit(session, alpha, "Where do I get my invoice?")
        await service.assign_next(EMBEDDER)
        gaps = list((await session.scalars(select(KnowledgeGap))).all())
        assert {gap.site_id for gap in gaps} == {alpha.id, beta.id}
        assert len(gaps) == 2


def test_queue_and_actions_are_per_site(client: TestClient) -> None:
    alpha, beta, staff, admin = _world(client)
    ages = [1, 2, 3, 4, 5]
    alpha_gap = seed_gap(alpha, QUESTION, [(QUESTION, age) for age in ages])
    beta_gap = seed_gap(beta, QUESTION, [(QUESTION, age) for age in ages])

    for site, gap in ((alpha, alpha_gap), (beta, beta_gap)):
        body = client.get("/api/knowledge-gaps", params={"site_id": str(site)}, headers=auth(staff))
        assert [item["id"] for item in body.json()["items"]] == [str(gap)]
        assert [item["site_id"] for item in body.json()["items"]] == [str(site)]
    everything = client.get("/api/knowledge-gaps", headers=auth(staff)).json()["items"]
    assert {item["id"] for item in everything} == {str(alpha_gap), str(beta_gap)}

    answered = client.post(
        f"/api/knowledge-gaps/{alpha_gap}/answer",
        headers=auth(admin),
        json={"kind": "canned", "shortcut": "invoice_download", "body": ANSWER},
    )
    assert answered.status_code == 204
    with db() as session:
        replies = list(session.scalars(select(CannedReply)).all())
        assert [(reply.site_id, reply.shortcut) for reply in replies] == [
            (alpha, "invoice_download")
        ]
    still_open = client.get(
        "/api/knowledge-gaps", params={"site_id": str(beta)}, headers=auth(staff)
    ).json()["items"]
    assert [item["id"] for item in still_open] == [str(beta_gap)]
    alpha_open = client.get(
        "/api/knowledge-gaps", params={"site_id": str(alpha)}, headers=auth(staff)
    ).json()["items"]
    assert alpha_open == []

    dismissed = client.post(f"/api/knowledge-gaps/{beta_gap}/dismiss", headers=auth(staff))
    assert dismissed.status_code == 204
    alpha_dismissed = client.get(
        "/api/knowledge-gaps",
        params={"site_id": str(alpha), "status": "dismissed"},
        headers=auth(staff),
    ).json()["items"]
    assert alpha_dismissed == []


def test_specialist_replies_come_only_from_the_gaps_own_chats(client: TestClient) -> None:
    alpha, beta, staff, _admin = _world(client)
    staff_id = insert_staff("jordan@example.local", "Jordan Lee", ALEX_PASSWORD)
    alpha_gap = seed_gap(alpha, QUESTION, [])
    beta_gap = seed_gap(beta, QUESTION, [])
    with db() as session:
        for site, gap, text in ((alpha, alpha_gap, "Alpha billing page"), (beta, beta_gap, "Beta")):
            chat = seed_conversation(session, site)
            seed_hit(session, site, gap, QUESTION, 1, chat)
            agent_message(session, site, chat, staff_id, f"{text}: open Billing and download it.")
    for gap, text in ((alpha_gap, "Alpha"), (beta_gap, "Beta")):
        replies = client.get(f"/api/knowledge-gaps/{gap}/replies", headers=auth(staff)).json()
        assert len(replies["items"]) == 1
        assert replies["items"][0].startswith(text)


def test_inbox_badge_reports_only_the_chats_own_gap(client: TestClient) -> None:
    alpha, beta, staff, _admin = _world(client)
    alpha_gap = seed_gap(alpha, QUESTION, [(QUESTION, age) for age in (1, 2, 3, 4)])
    beta_gap = seed_gap(beta, QUESTION, [(QUESTION, age) for age in (1, 2, 3, 4)])
    with db() as session:
        alpha_chat = seed_conversation(session, alpha)
        seed_hit(session, alpha, alpha_gap, QUESTION, 1, alpha_chat)
        beta_chat = seed_conversation(session, beta)
        seed_hit(session, beta, beta_gap, QUESTION, 1, beta_chat)
    for chat, gap in ((alpha_chat, alpha_gap), (beta_chat, beta_gap)):
        badge = client.get(f"/api/conversations/{chat}/knowledge-gap", headers=auth(staff)).json()
        assert badge["gap"]["id"] == str(gap)
        assert badge["gap"]["conversations"] == 5


def test_closest_entries_for_a_gap_ignore_the_other_site(client: TestClient) -> None:
    """Identical text and an identical vector on the other site must not be suggested."""
    alpha, beta, staff, _admin = _world(client)
    vector = EMBEDDER.vector(QUESTION)
    gap = seed_gap(alpha, QUESTION, [(QUESTION, age) for age in (1, 2, 3, 4, 5)])
    app.dependency_overrides[get_embedder] = lambda: EMBEDDER
    try:
        with db() as session:
            session.add(
                CannedReply(
                    site_id=beta,
                    shortcut="invoice_copy",
                    body=ANSWER,
                    embedding=vector,
                    embedder_id=EMBEDDER.embedder_id,
                )
            )
            seed_chunk(session, beta, QUESTION, ANSWER, vector)
        none = client.get(f"/api/knowledge-gaps/{gap}/similar", headers=auth(staff)).json()
        assert none == {"canned": None, "knowledge": None}
        with db() as session:
            session.add(
                CannedReply(
                    site_id=alpha,
                    shortcut="invoice_copy",
                    body=ANSWER,
                    embedding=vector,
                    embedder_id=EMBEDDER.embedder_id,
                )
            )
            seed_chunk(session, alpha, QUESTION, ANSWER, vector)
        found = client.get(f"/api/knowledge-gaps/{gap}/similar", headers=auth(staff)).json()
    finally:
        app.dependency_overrides.pop(get_embedder, None)
    assert found["canned"]["site_id"] == str(alpha)
    assert found["knowledge"]["title"] == QUESTION
