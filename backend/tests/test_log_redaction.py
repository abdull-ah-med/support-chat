import json
import uuid
from types import SimpleNamespace

from fastapi.testclient import TestClient
from structlog.testing import capture_logs

from app.db import session_maker
from app.llm.bot_responder import BotResponder
from app.services.conversation_service import ConversationService
from tests.bot_fixtures import (
    ADA_EMAIL,
    EASY_BODY,
    EASY_TITLE,
    insert_bot_conversation,
    insert_site,
)
from tests.fake_anthropic import FakeAnthropic
from tests.ws_helpers import (
    ALEX_EMAIL,
    ALEX_PASSWORD,
    DEMO_PUBLIC_KEY,
    DEMO_SITE_KEY,
    HOST_ORIGIN,
    WIDGET_ORIGIN,
    auth_visitor,
    collect_until,
    insert_staff,
    post_bootstrap,
)
from tests.ws_helpers import (
    insert_site as insert_site_sync,
)

DOT_BODY = "How fast are DOT results?"
FIXTURE_TOKEN = "resume-token-fixture-value"


async def test_visitor_message_log_has_ids_not_body_or_email(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "samplesite", "SampleSite")
        visitor, conversation = await insert_bot_conversation(session, site)
        conversation.state = "queued"
        await session.commit()
        conversation_id = conversation.id
        visitor_id = visitor.id
    with capture_logs() as logs:
        async with session_maker()() as session:
            await ConversationService(session).visitor_message(
                conversation_id,
                visitor_id,
                HOST_ORIGIN,
                uuid.uuid4(),
                DOT_BODY,
            )
    serialized = json.dumps(logs)
    assert DOT_BODY not in serialized
    assert ADA_EMAIL not in serialized
    assert "Ada Lopez" not in serialized
    assert any(entry.get("conversation_id") == str(conversation_id) for entry in logs)
    assert any(entry.get("site_key") == "samplesite" for entry in logs)
    assert any(entry.get("role") == "visitor" and entry.get("length") == 25 for entry in logs)


def _turn_and_unit():
    from app.services.grounded_response import EvidenceUnit, TurnContext

    unit = EvidenceUnit(
        id=uuid.uuid4(),
        canonical_question=None,
        aliases=(),
        topic_label=EASY_TITLE,
        answer_verbatim=EASY_BODY,
        source_title=EASY_TITLE,
        source_url="https://example.test/timing",
        snapshot_id=uuid.uuid4(),
    )
    return TurnContext(visitor_text=DOT_BODY, evidence=[unit], site_name="SampleSite"), unit


async def test_provider_exception_log_omits_prompt(migrated_db) -> None:
    turn, unit = _turn_and_unit()
    with capture_logs() as logs:
        draft = await BotResponder(client=FakeAnthropic(fail={"draft"})).generate_grounded_draft(
            turn, [unit]
        )
    serialized = json.dumps(logs)
    assert draft is None
    assert DOT_BODY not in serialized
    assert EASY_BODY not in serialized
    assert "scripted draft failure" not in serialized
    assert any(entry.get("error_class") == "TimeoutError" for entry in logs)


async def test_sdk_argument_mismatch_log_names_only_the_invalid_parameter(migrated_db) -> None:
    class Messages:
        async def create(self, **_kwargs):
            raise TypeError(
                "AsyncMessages.create() got an unexpected keyword argument 'temperature'"
            )

    turn, unit = _turn_and_unit()
    with capture_logs() as logs:
        await BotResponder(client=SimpleNamespace(messages=Messages())).generate_grounded_draft(
            turn, [unit]
        )

    matching = [entry for entry in logs if entry.get("event") == "provider_error"]
    assert len(matching) == 1
    assert matching[0]["error_class"] == "TypeError"
    assert matching[0]["error_code"] == "sdk_argument_mismatch"
    assert matching[0]["invalid_parameter"] == "temperature"
    assert "exception" not in matching[0]


def test_malformed_frame_log_omits_payload(client: TestClient) -> None:
    insert_site_sync(DEMO_SITE_KEY, "Demo", DEMO_PUBLIC_KEY)
    token = post_bootstrap(client).json()["bootstrap_token"]
    with client.websocket_connect("/ws/visitor", headers={"Origin": WIDGET_ORIGIN}) as ws:
        auth_visitor(ws, token)
        collect_until(ws, lambda frames: any(frame.get("type") == "state" for frame in frames))
        with capture_logs() as logs:
            ws.send_text("{not-json " + DOT_BODY + " " + FIXTURE_TOKEN)
            collect_until(ws, lambda frames: any(frame.get("type") == "error" for frame in frames))
    serialized = json.dumps(logs)
    assert DOT_BODY not in serialized
    assert FIXTURE_TOKEN not in serialized
    assert any(entry.get("event") == "malformed_frame" for entry in logs)


async def test_wakeup_catch_up_failure_logs_conversation_id(migrated_db, monkeypatch) -> None:
    from app.chat.connection_manager import connection_manager
    from app.services.conversation_service import ConversationService

    conversation_id = uuid.uuid4()

    async def boom(self, *args, **kwargs):
        raise RuntimeError("replay failed")

    monkeypatch.setattr(ConversationService, "replay", boom)
    with capture_logs() as logs:
        await connection_manager.deliver_wakeup(
            {"conversation_id": str(conversation_id), "site_key": "samplesite"}
        )
    matching = [entry for entry in logs if entry.get("event") == "wakeup_catch_up_failed"]
    assert len(matching) == 1
    assert matching[0]["conversation_id"] == str(conversation_id)
    assert matching[0]["error_class"] == "RuntimeError"
    serialized = json.dumps(logs)
    assert "replay failed" not in serialized


def test_session_and_inbox_responses_are_no_store(client: TestClient) -> None:
    insert_staff(ALEX_EMAIL, "Alex Morgan", ALEX_PASSWORD)
    login = client.post(
        "/auth/login",
        json={"email": ALEX_EMAIL, "password": ALEX_PASSWORD},
        headers={"Origin": "http://localhost:3000"},
    )
    inbox = client.get(
        "/api/conversations",
        headers={"Authorization": f"Bearer {login.json()['access_token']}"},
    )
    assert login.headers["cache-control"] == "no-store"
    assert inbox.headers["cache-control"] == "no-store"
