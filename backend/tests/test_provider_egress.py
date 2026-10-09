"""What leaves for the model provider: redacted visitor text, public knowledge contacts kept."""

import json

from app.db import session_maker
from app.services.pii_redactor import REDACTED, redact_for_model
from tests.bot_fixtures import EASY_BODY, EASY_TITLE
from tests.fake_anthropic import Draft, FakeAnthropic
from tests.turn_helpers import open_chat

PRIOR_VISITOR_LINE = "My email is ada@example.com and phone is (555) 867-5309"
CURRENT_VISITOR_LINE = "How fast are results?"
KB_CONTACT_SNIPPET = "Call billing at support@example.test or 800-555-0199."


def test_redact_for_model_masks_email_phone_and_existing_pii_literals() -> None:
    raw = (
        "SSN 123-45-6789, DL A1234567, plate AB-1234, DOB 1990-01-15, "
        "email ada@example.com, phone (555) 867-5309 and 8005550199"
    )
    redacted = redact_for_model(raw)
    for secret in (
        "123-45-6789",
        "A1234567",
        "AB-1234",
        "1990-01-15",
        "ada@example.com",
        "(555) 867-5309",
        "8005550199",
    ):
        assert secret not in redacted
    assert redacted.count(REDACTED) >= 7


async def test_every_request_of_a_turn_carries_only_redacted_visitor_text(migrated_db) -> None:
    client = FakeAnthropic(
        draft=Draft("Most negative results are reported within 24-48 hours.", EASY_BODY),
        understand=[{}, {"relation": "continuation"}],
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[(EASY_TITLE, EASY_BODY)])
        await chat.say(PRIOR_VISITOR_LINE)
        await chat.say(CURRENT_VISITOR_LINE)

    assert client.calls
    for call in client.calls:
        serialized = json.dumps(call["messages"], ensure_ascii=False)
        assert "ada@example.com" not in serialized, call["stage"]
        assert "867-5309" not in serialized, call["stage"]
    # The first message reaches the second turn's understanding only as a redacted history entry.
    second = json.loads(client.stage_calls("understand_turn")[1]["messages"][0]["content"])
    assert second["history"][0] == {
        "role": "user",
        "content": redact_for_model(PRIOR_VISITOR_LINE),
    }
    assert REDACTED in second["history"][0]["content"]
    assert second["latest_message"] == CURRENT_VISITOR_LINE


async def test_public_knowledge_contacts_reach_the_provider_unmasked(migrated_db) -> None:
    client = FakeAnthropic(draft=Draft("Call billing at support@example.test.", KB_CONTACT_SNIPPET))
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[("Billing", KB_CONTACT_SNIPPET)])
        await chat.say("billing contact")

    serialized = json.dumps(client.stage_calls("draft")[0]["messages"])
    assert "support@example.test" in serialized
    assert "800-555-0199" in serialized


async def test_history_replays_staff_and_bot_turns_with_roles_in_the_draft_request(
    migrated_db,
) -> None:
    client = FakeAnthropic(
        draft=Draft("Most negative results are reported within 24-48 hours.", EASY_BODY),
        understand=[{}, {"relation": "continuation"}],
    )
    async with session_maker()() as session:
        chat = await open_chat(session, client, chunks=[(EASY_TITLE, EASY_BODY)])
        await chat.say("How fast are results?")
        await chat.say("And how fast are results again?")

    messages = client.stage_calls("draft")[1]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert isinstance(messages[-1]["content"], list)
