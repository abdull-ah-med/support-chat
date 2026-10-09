"""Delivery reliability: poison jobs, Retry-After, headers and sessions on other devices."""

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import select

from app.db import session_maker
from app.services.web_push import PUSH_TTL_SECONDS, PushStatus, WebPushSender
from app.settings import get_settings
from tests.test_notifications import auth, escalate, make_chat
from tests.test_push_notifications import (
    RecordingSender,
    deliver,
    pending_count,
    push_config,  # noqa: F401
    subscription,
    subscription_count,
)
from tests.ws_helpers import insert_site

pytestmark = pytest.mark.usefixtures("push_config")


class RaisingSender(RecordingSender):
    def __init__(self, error):
        super().__init__(())
        self.error = error

    async def send(self, device, payload):
        raise self.error


def queue_one(client, headers, endpoint="https://fcm.googleapis.com/fcm/send/device-a"):
    site = insert_site("easy", "SampleSite", "key")
    client.post(
        "/api/notifications/push/subscriptions", headers=headers, json=subscription(endpoint)
    )
    chat_id, visitor_id = client.portal.call(make_chat, site)
    client.portal.call(escalate, chat_id, visitor_id)


async def next_attempt():
    from app.models.push_subscription import PushDelivery

    async with session_maker()() as session:
        return (await session.scalars(select(PushDelivery))).one().next_attempt_at


@pytest.mark.parametrize("error", [ValueError("bad key"), RuntimeError("boom")])
def test_unexpected_send_error_does_not_wedge_the_queue(client, error):
    _, headers = auth(client)
    queue_one(client, headers)
    assert client.portal.call(deliver, RaisingSender(error)) is True
    assert client.portal.call(pending_count) == 0
    assert client.portal.call(deliver, RecordingSender()) is False


def test_retry_after_is_honoured_and_capped(client):
    _, headers = auth(client)
    queue_one(client, headers)
    now = datetime.now(UTC)
    sender = RecordingSender((PushStatus(429, retry_after=300),))
    assert client.portal.call(deliver, sender, now) is True
    assert client.portal.call(next_attempt) >= now + timedelta(seconds=300)
    assert client.portal.call(deliver, RecordingSender(), now + timedelta(seconds=60)) is False
    sender = RecordingSender((201,))
    assert client.portal.call(deliver, sender, now + timedelta(seconds=301)) is True
    assert client.portal.call(pending_count) == 0


@pytest.mark.parametrize("status", [400])
def test_permanent_rejection_is_dropped_but_keeps_the_device(client, status):
    _, headers = auth(client)
    queue_one(client, headers)
    assert client.portal.call(deliver, RecordingSender((status,))) is True
    assert client.portal.call(pending_count) == 0
    assert client.portal.call(subscription_count) == 1


@pytest.mark.parametrize("status", [408, 500])
def test_transient_statuses_are_retried(client, status):
    _, headers = auth(client)
    queue_one(client, headers)
    assert client.portal.call(deliver, RecordingSender((status,))) is True
    assert client.portal.call(pending_count) == 1


def test_logout_on_one_device_does_not_silence_the_users_other_devices(client):
    _, headers = auth(client)
    queue_one(client, headers, "https://fcm.googleapis.com/fcm/send/laptop")
    client.post(
        "/api/notifications/push/subscriptions",
        headers=headers,
        json=subscription("https://fcm.googleapis.com/fcm/send/phone"),
    )
    # This device unsubscribes itself (as the web client does) and then signs out.
    client.request(
        "DELETE",
        "/api/notifications/push/subscriptions",
        headers=headers,
        json={"endpoint": "https://fcm.googleapis.com/fcm/send/laptop"},
    )
    csrf = client.cookies.get("supportchat_csrf")
    assert client.post("/auth/logout", headers={"X-CSRF-Token": csrf}).status_code == 200
    site = insert_site("other", "Other", "other-key")
    chat_id, visitor_id = client.portal.call(make_chat, site)
    client.portal.call(escalate, chat_id, visitor_id)
    sender = RecordingSender()
    while client.portal.call(deliver, sender):
        pass
    assert len(sender.delivered) >= 1


def test_password_change_still_revokes_every_device(client):
    from tests.ws_helpers import ALEX_PASSWORD

    _, headers = auth(client)
    queue_one(client, headers)
    csrf = client.cookies.get("supportchat_csrf")
    response = client.post(
        "/auth/change-password",
        headers={**headers, "X-CSRF-Token": csrf},
        json={"current_password": ALEX_PASSWORD, "new_password": "a-brand-new-passphrase-9"},
    )
    assert response.status_code == 200, response.text
    sender = RecordingSender(())
    client.portal.call(deliver, sender)
    assert sender.delivered == []


def test_sender_asks_for_high_urgency_and_a_useful_ttl_and_reads_retry_after(monkeypatch):
    seen = []

    def reply(request):
        seen.append(request)
        return httpx.Response(429, headers={"Retry-After": "42"})

    async def send():
        details = subscription()
        device = SimpleNamespace(endpoint=details["endpoint"], **details["keys"])
        async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
            return await WebPushSender(get_settings(), client).send(device, {"title": "x"})

    status = asyncio.run(send())
    assert status == 429 and status.retry_after == 42
    assert seen[0].headers["Urgency"] == "high"
    assert seen[0].headers["TTL"] == str(PUSH_TTL_SECONDS) and PUSH_TTL_SECONDS >= 600
