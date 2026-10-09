"""Staff APIs: every list, search and feed can be scoped to one site, and two sites that use
the same words, shortcuts and aliases never see each other's rows.
"""

import csv
import io
import uuid
from functools import partial

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db import session_maker
from app.models.canned_reply import CannedReply
from app.models.conversation import Conversation
from app.models.kb_snapshot import KbSnapshot
from app.models.knowledge_gap import KnowledgeGap
from app.models.message import Message
from app.models.notification import Notification, NotificationPushPreference
from app.models.push_subscription import PushDelivery, PushSubscription
from app.models.site import Site
from app.models.user import User
from app.models.visitor import Visitor
from app.services.handoff_service import HandoffService, HandoffTrigger
from tests.knowledge_gap_fixtures import seed_chunk
from tests.site_isolation_helpers import (
    BETA_ONLY_ANSWER,
    BETA_ONLY_QUESTION,
    SHARED_ANSWER,
    SHARED_QUESTION,
    add_live_chunk,
    seed_two_sites,
)
from tests.ws_helpers import (
    ALEX_EMAIL,
    ALEX_NAME,
    ALEX_PASSWORD,
    HOST_ORIGIN,
    insert_site,
    insert_staff,
    login_staff,
    sync_session,
)

ADMIN_EMAIL = "admin@example.local"
TOKEN = "zebra-token"


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _world(client: TestClient):
    alpha = insert_site("alpha", "Alpha Co", "a" * 64, [HOST_ORIGIN])
    beta = insert_site("beta", "Beta Co", "b" * 64, [HOST_ORIGIN])
    insert_staff(ALEX_EMAIL, ALEX_NAME, ALEX_PASSWORD)
    insert_staff(ADMIN_EMAIL, "Admin", ALEX_PASSWORD, is_admin=True)
    staff = _headers(login_staff(client))
    admin = _headers(login_staff(client, ADMIN_EMAIL, ALEX_PASSWORD))
    return alpha, beta, staff, admin


def _staff_only(client: TestClient) -> dict[str, str]:
    insert_staff(ALEX_EMAIL, ALEX_NAME, ALEX_PASSWORD)
    return _headers(login_staff(client))


async def _chat(site_id, *, body: str, state: str = "bot", submitted: bool = True):
    async with session_maker()() as session:
        visitor = Visitor(
            site_id=site_id,
            resume_token_hash=uuid.uuid4().hex,
            name=f"Visitor {TOKEN}",
            email=f"{TOKEN}@example.com",
        )
        session.add(visitor)
        await session.flush()
        extra = (
            {"prechat_submission_id": uuid.uuid4(), "prechat_payload_hash": "a" * 64}
            if submitted
            else {}
        )
        chat = Conversation(site_id=site_id, visitor_id=visitor.id, state=state, **extra)
        session.add(chat)
        await session.flush()
        session.add(
            Message(
                conversation_id=chat.id,
                site_id=site_id,
                role="visitor",
                body=body,
                client_message_id=uuid.uuid4(),
            )
        )
        await session.commit()
        return chat.id


# --- canned replies ------------------------------------------------------------------------


def test_two_sites_can_use_the_same_shortcut_and_alias(client: TestClient) -> None:
    alpha, beta, _staff, admin = _world(client)
    for site, body in ((alpha, "Alpha hours"), (beta, "Beta hours")):
        created = client.post(
            "/api/canned-replies",
            headers=admin,
            json={
                "site_id": str(site),
                "shortcut": "hours",
                "aliases": ["opening"],
                "body": body,
            },
        )
        assert created.status_code == 201, created.text
    again = client.post(
        "/api/canned-replies",
        headers=admin,
        json={"site_id": str(alpha), "shortcut": "opening", "body": "Clash on the alias"},
    )
    assert again.status_code == 409
    for site, body, other in ((alpha, "Alpha hours", "Beta"), (beta, "Beta hours", "Alpha")):
        listed = client.get("/api/canned-replies", params={"site_id": str(site)}, headers=admin)
        assert [row["body"] for row in listed.json()["items"]] == [body]
        assert other not in listed.text


def test_effective_list_has_the_sites_replies_plus_general_only(client: TestClient) -> None:
    alpha, beta, _staff, admin = _world(client)
    for site_id, shortcut in ((None, "general_note"), (alpha, "alpha_note"), (beta, "beta_note")):
        client.post(
            "/api/canned-replies",
            headers=admin,
            json={
                "site_id": None if site_id is None else str(site_id),
                "shortcut": shortcut,
                "body": f"Body of {shortcut}",
            },
        )
    seen = {}
    for name, site in (("alpha", alpha), ("beta", beta)):
        rows = client.get("/api/canned-replies", params={"site_id": str(site)}, headers=admin)
        seen[name] = {row["shortcut"]: row["scope"] for row in rows.json()["items"]}
    assert seen["alpha"] == {"general_note": "general", "alpha_note": "website"}
    assert seen["beta"] == {"general_note": "general", "beta_note": "website"}


def test_a_site_reply_shadows_general_only_on_its_own_site(client: TestClient) -> None:
    alpha, beta, _staff, admin = _world(client)
    for site_id, body in ((None, "General hours"), (alpha, "Alpha hours")):
        client.post(
            "/api/canned-replies",
            headers=admin,
            json={
                "site_id": None if site_id is None else str(site_id),
                "shortcut": "hours",
                "body": body,
            },
        )
    bodies = {}
    for name, site in (("alpha", alpha), ("beta", beta)):
        rows = client.get("/api/canned-replies", params={"site_id": str(site)}, headers=admin)
        bodies[name] = [row["body"] for row in rows.json()["items"]]
    assert bodies == {"alpha": ["Alpha hours"], "beta": ["General hours"]}


def test_moving_a_reply_to_another_site_drops_it_from_the_first(client: TestClient) -> None:
    alpha, beta, _staff, admin = _world(client)
    created = client.post(
        "/api/canned-replies",
        headers=admin,
        json={"site_id": str(alpha), "shortcut": "moving", "body": "Mover"},
    ).json()
    moved = client.patch(
        f"/api/canned-replies/{created['id']}", headers=admin, json={"site_id": str(beta)}
    )
    assert moved.status_code == 200
    on_alpha = client.get("/api/canned-replies", params={"site_id": str(alpha)}, headers=admin)
    on_beta = client.get("/api/canned-replies", params={"site_id": str(beta)}, headers=admin)
    assert on_alpha.json()["items"] == []
    assert [row["shortcut"] for row in on_beta.json()["items"]] == ["moving"]


# --- conversations -------------------------------------------------------------------------


def test_inbox_lists_and_counts_are_per_site(client: TestClient) -> None:
    alpha, beta, staff, _admin = _world(client)
    alpha_chat = client.portal.call(partial(_chat, alpha, body=SHARED_QUESTION))
    beta_chat = client.portal.call(partial(_chat, beta, body=SHARED_QUESTION))
    client.portal.call(partial(_chat, beta, body=SHARED_QUESTION, state="queued"))
    for site, chat, expected in ((alpha, alpha_chat, 1), (beta, beta_chat, 2)):
        listed = client.get("/api/conversations", params={"site_id": str(site)}, headers=staff)
        body = listed.json()
        assert {item["site_id"] for item in body["items"]} == {str(site)}
        assert str(chat) in {item["id"] for item in body["items"]}
        assert len(body["items"]) == expected
        assert sum(body["counts"].values()) == expected
    everything = client.get("/api/conversations", headers=staff).json()
    assert sum(everything["counts"].values()) == 3
    assert {row["id"] for row in everything["sites"]} == {str(alpha), str(beta)}
    unknown = client.get("/api/conversations", params={"site_id": str(uuid.uuid4())}, headers=staff)
    assert unknown.status_code == 400


def test_data_submissions_and_export_filter_by_site(client: TestClient) -> None:
    alpha, beta, staff, _admin = _world(client)
    alpha_chat = client.portal.call(partial(_chat, alpha, body=SHARED_QUESTION))
    beta_chat = client.portal.call(partial(_chat, beta, body=SHARED_QUESTION))
    for site, chat, other in ((alpha, alpha_chat, beta_chat), (beta, beta_chat, alpha_chat)):
        listed = client.get(
            "/api/conversations/submissions", params={"site_id": str(site)}, headers=staff
        ).json()
        assert [item["id"] for item in listed["items"]] == [str(chat)]
        assert str(other) not in {item["id"] for item in listed["items"]}
    everything = client.get("/api/conversations/submissions", headers=staff).json()
    assert {item["id"] for item in everything["items"]} == {str(alpha_chat), str(beta_chat)}

    exported = client.post(
        "/api/conversations/submissions/export",
        headers=staff,
        json={"columns": ["Site"], "site_id": str(alpha)},
    )
    assert exported.status_code == 200
    rows = list(csv.reader(io.StringIO(exported.text)))
    assert [row[0] for row in rows[1:]] == ["Alpha Co"]


def test_inbox_detail_of_one_site_never_carries_another_sites_data(client: TestClient) -> None:
    alpha, beta, staff, _admin = _world(client)
    alpha_chat = client.portal.call(partial(_chat, alpha, body="Alpha question"))
    client.portal.call(partial(_chat, beta, body="Beta question"))
    detail = client.get(f"/api/conversations/{alpha_chat}", headers=staff).json()
    assert detail["site_id"] == str(alpha)
    assert detail["site_name"] == "Alpha Co"
    assert [message["body"] for message in detail["messages"]] == ["Alpha question"]


async def _open_handoff_naming(chat_id, chunk_id) -> None:
    from app.models.handoff_context import HandoffContext

    async with session_maker()() as session:
        await HandoffService(session).open_handoff(
            HandoffTrigger(
                conversation_id=chat_id, reason="visitor_request", original_question="Help"
            )
        )
        row = await session.scalar(
            select(HandoffContext).where(HandoffContext.conversation_id == chat_id)
        )
        row.candidate_unit_ids = [chunk_id]
        await session.commit()


async def _beta_chunk(beta_id) -> uuid.UUID:
    async with session_maker()() as session:
        beta = await session.get(Site, beta_id)
        chunk = await add_live_chunk(session, beta, BETA_ONLY_QUESTION, BETA_ONLY_ANSWER)
        await session.commit()
        return chunk.id


def test_handoff_candidates_never_label_another_sites_chunk(client: TestClient) -> None:
    alpha, beta, staff, _admin = _world(client)
    beta_chunk = client.portal.call(_beta_chunk, beta)
    alpha_chat = client.portal.call(partial(_chat, alpha, body="Help"))
    client.portal.call(_open_handoff_naming, alpha_chat, beta_chunk)
    handoff = client.get(f"/api/conversations/{alpha_chat}/handoff", headers=staff)
    assert handoff.status_code == 200
    assert [item["canonical_question"] for item in handoff.json()["candidates"]] == [None]
    assert BETA_ONLY_QUESTION not in handoff.text


# --- workspace search ----------------------------------------------------------------------


def _seed_searchable(alpha: uuid.UUID, beta: uuid.UUID, client: TestClient):
    chats = {
        "alpha": client.portal.call(partial(_chat, alpha, body=f"Question about {TOKEN}")),
        "beta": client.portal.call(partial(_chat, beta, body=f"Question about {TOKEN}")),
    }
    session = next(sync_session())
    try:
        for site in (alpha, beta):
            session.add(
                CannedReply(site_id=site, shortcut=TOKEN, aliases=[], body=f"Reply about {TOKEN}")
            )
            session.add(
                KnowledgeGap(site_id=site, question=f"{TOKEN} question", status="dismissed")
            )
            seed_chunk(session, site, f"{TOKEN} FAQ", f"The {TOKEN} answer.", [0.0] * 1536)
        session.add(
            CannedReply(site_id=None, shortcut="general_zebra", body=f"General {TOKEN} script")
        )
        session.commit()
    finally:
        session.close()
    return chats


def _search(client, headers, **extra):
    response = client.post(
        "/api/search", headers=headers, json={"query": TOKEN, "screen": "inbox", **extra}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    return body["current"] + body["other"]


def test_search_without_a_site_finds_both_and_labels_each(client: TestClient) -> None:
    alpha, beta, staff, _admin = _world(client)
    _seed_searchable(alpha, beta, client)
    items = _search(client, staff)
    descriptions = " ".join(item["description"] for item in items)
    assert "Alpha Co" in descriptions and "Beta Co" in descriptions


def test_search_scoped_to_a_site_returns_only_that_sites_records(client: TestClient) -> None:
    alpha, beta, staff, admin = _world(client)
    chats = _seed_searchable(alpha, beta, client)
    for caller in (staff, admin):
        for site, name, other_name, other_id, other_chat in (
            (alpha, "Alpha Co", "Beta Co", beta, chats["beta"]),
            (beta, "Beta Co", "Alpha Co", alpha, chats["alpha"]),
        ):
            items = _search(client, caller, site_id=str(site))
            assert items, "the scoped search should still find the site's own records"
            for item in items:
                assert other_name not in item["description"], item
                assert str(other_id) not in item["href"], item
                assert str(other_chat) not in item["href"], item
                assert item["kind"] != "log", item
            kinds = {item["kind"] for item in items}
            assert {"conversation", "response", "gap", "answer"} <= kinds
            assert any(name in item["description"] for item in items)


def test_scoped_search_keeps_general_replies_and_omits_logs(client: TestClient) -> None:
    alpha, beta, _staff, admin = _world(client)
    _seed_searchable(alpha, beta, client)
    session = next(sync_session())
    try:
        from app.models.app_log import AppLog

        session.add(AppLog(level="error", source="backend", event=f"{TOKEN}_event", message=TOKEN))
        session.commit()
    finally:
        session.close()
    unscoped = _search(client, admin)
    scoped = _search(client, admin, site_id=str(alpha))
    assert "log" in {item["kind"] for item in unscoped}
    assert "log" not in {item["kind"] for item in scoped}
    assert any(item["kind"] == "response" and "general_zebra" in item["title"] for item in scoped)


# --- notifications and push ----------------------------------------------------------------


async def _escalate(chat_id):
    async with session_maker()() as session:
        await HandoffService(session).open_handoff(
            HandoffTrigger(
                conversation_id=chat_id, reason="visitor_request", original_question="Help"
            )
        )
        await session.commit()


def test_notification_feed_can_be_scoped_to_one_site(client: TestClient) -> None:
    alpha, beta, staff, _admin = _world(client)
    for site in (alpha, beta, beta):
        chat = client.portal.call(partial(_chat, site, body="Help", submitted=False))
        client.portal.call(_escalate, chat)
    everything = client.get("/api/notifications", headers=staff).json()
    assert everything["unread_count"] == 3
    for site, expected in ((alpha, 1), (beta, 2)):
        feed = client.get("/api/notifications", params={"site_id": str(site)}, headers=staff)
        body = feed.json()
        assert {item["site_id"] for item in body["items"]} == {str(site)}
        assert len(body["items"]) == expected
        assert body["unread_count"] == expected
        assert {ctx["site_id"] for ctx in body["unread_conversation_context"].values()} == {
            str(site)
        }
        assert body["latest_id"] == max(item["id"] for item in body["items"])


def test_push_preferences_and_deliveries_are_per_site(client: TestClient) -> None:
    alpha, beta, _staff, _admin = _world(client)
    session = next(sync_session())
    try:
        user_id = session.scalar(select(User.id).where(User.email == ALEX_EMAIL))
        session.add(
            PushSubscription(
                user_id=user_id,
                token_version=0,
                endpoint="https://push.example.test/device",
                p256dh="p",
                auth="a",
            )
        )
        # Push switched off for alpha only.
        session.add(
            NotificationPushPreference(
                user_id=user_id, site_id=alpha, enabled=False, scenarios=["needs_attention"]
            )
        )
        session.commit()
    finally:
        session.close()
    for site in (alpha, beta):
        chat = client.portal.call(partial(_chat, site, body="Help", submitted=False))
        client.portal.call(_escalate, chat)

    session = next(sync_session())
    try:
        delivered_sites = list(
            session.scalars(
                select(Notification.site_id).join(
                    PushDelivery, PushDelivery.notification_id == Notification.id
                )
            ).all()
        )
    finally:
        session.close()
    assert delivered_sites == [beta]


# --- knowledge sources ---------------------------------------------------------------------


def test_source_lists_and_pages_are_per_site(client: TestClient) -> None:
    staff = _staff_only(client)

    async def seed():
        async with session_maker()() as session:
            return await seed_two_sites(session)

    seeded = client.portal.call(seed)
    for site, own, foreign in (
        (seeded.alpha, seeded.alpha_shared, seeded.beta_shared),
        (seeded.beta, seeded.beta_shared, seeded.alpha_shared),
    ):
        listed = client.get(f"/api/sites/{site.id}/kb-sources", headers=staff).json()["items"]
        assert len(listed) == 2
        assert {row["site_id"] for row in listed} == {str(site.id)}
        page_ids = set()
        for row in listed:
            pages = client.get(f"/api/kb-sources/{row['id']}/pages", headers=staff).json()
            page_ids |= {item["id"] for item in pages["items"]}
        assert str(own.page_id) in page_ids
        assert str(foreign.page_id) not in page_ids


def test_snapshot_diff_refuses_a_snapshot_of_another_source(client: TestClient) -> None:
    staff = _staff_only(client)

    async def seed():
        async with session_maker()() as session:
            seeded = await seed_two_sites(session)
            alpha_snapshot = await session.get(KbSnapshot, seeded.alpha_shared.snapshot_id)
            return (
                alpha_snapshot.source_id,
                seeded.alpha_shared.snapshot_id,
                seeded.beta_shared.snapshot_id,
            )

    source_id, own_snapshot, foreign_snapshot = client.portal.call(seed)
    allowed = client.get(
        f"/api/kb-sources/{source_id}/diff", params={"to": str(own_snapshot)}, headers=staff
    )
    assert allowed.status_code == 200
    for params in ({"to": str(foreign_snapshot)}, {"from": str(foreign_snapshot)}):
        denied = client.get(f"/api/kb-sources/{source_id}/diff", params=params, headers=staff)
        assert denied.status_code == 404
        assert BETA_ONLY_ANSWER not in denied.text and SHARED_ANSWER not in denied.text


def test_articles_and_pages_are_not_addressable_through_another_site(client: TestClient) -> None:
    alpha, beta, staff, admin = _world(client)
    created = client.post(
        f"/api/sites/{alpha}/articles", headers=admin, json={"title": "Same title", "body": "A"}
    )
    other = client.post(
        f"/api/sites/{beta}/articles", headers=admin, json={"title": "Same title", "body": "B"}
    )
    assert created.status_code == 201 and other.status_code == 201
    for site, body in ((alpha, "A"), (beta, "B")):
        listed = client.get(f"/api/sites/{site}/articles", headers=staff).json()["items"]
        assert [(row["site_id"], row["body"]) for row in listed] == [(str(site), body)]


# --- rate limits ---------------------------------------------------------------------------


async def test_visitor_create_budget_is_per_site(migrated_db) -> None:
    from app.services.rate_limit import RateLimiter, RateLimitExceeded
    from app.settings import get_settings

    limiter = RateLimiter()
    alpha, beta = str(uuid.uuid4()), str(uuid.uuid4())
    budget = get_settings().rate_visitor_create
    for _ in range(budget):
        await limiter.hit_visitor_create("203.0.113.9", alpha)
    try:
        await limiter.hit_visitor_create("203.0.113.9", alpha)
        raise AssertionError("alpha should be rate limited")
    except RateLimitExceeded:
        pass
    await limiter.hit_visitor_create("203.0.113.9", beta)


# --- visitor blocks ------------------------------------------------------------------------


async def _is_blocked(site_id, email: str) -> bool:
    from app.repositories.visitor_block_repo import VisitorBlockRepository

    async with session_maker()() as session:
        return await VisitorBlockRepository(session).is_blocked(
            site_id, ip=None, email=email, phone=None
        )


def test_a_block_on_one_site_does_not_apply_to_another(client: TestClient) -> None:
    alpha, beta, staff, _admin = _world(client)
    created = client.post(
        "/api/visitor-blocks",
        headers=staff,
        json={"site_id": str(alpha), "email": "spam@example.com"},
    )
    assert created.status_code == 201, created.text
    assert client.portal.call(_is_blocked, alpha, "spam@example.com") is True
    assert client.portal.call(_is_blocked, beta, "spam@example.com") is False
    scoped = {
        name: client.get(
            "/api/visitor-blocks", params={"site_id": str(site)}, headers=staff
        ).json()["items"]
        for name, site in (("alpha", alpha), ("beta", beta))
    }
    assert [item["site_id"] for item in scoped["alpha"]] == [str(alpha)]
    assert scoped["beta"] == []
    assert len(client.get("/api/visitor-blocks", headers=staff).json()["items"]) == 1


def test_two_sites_can_each_add_the_same_text_source(client: TestClient) -> None:
    alpha, beta, staff, admin = _world(client)
    payload = {"kind": "text", "title": "Shared title", "body": "Shared body text."}
    created = {
        name: client.post(f"/api/sites/{site}/kb-sources", headers=admin, json=payload)
        for name, site in (("alpha", alpha), ("beta", beta))
    }
    assert {r.status_code for r in created.values()} == {201}
    for name, site in (("alpha", alpha), ("beta", beta)):
        listed = client.get(f"/api/sites/{site}/kb-sources", headers=staff).json()["items"]
        assert [row["id"] for row in listed] == [created[name].json()["id"]]
        assert {row["site_id"] for row in listed} == {str(site)}
