"""Per-site assistant profile (scope, hand-off topics, language) and free-text intents."""

import uuid

import pytest
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

from alembic import command
from app.models.user import User
from app.security.passwords import hash_password
from tests.conftest import TEST_DATABASE_URL
from tests.ws_helpers import insert_site, login_staff, sync_session

ADMIN_EMAIL = "admin@example.local"
ADMIN_PASSWORD = "secret"
SCOPE = "Opening hours, deliveries and returns for the shop."


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _login_admin(client: TestClient) -> str:
    session = next(sync_session())
    try:
        session.add(
            User(
                email=ADMIN_EMAIL,
                display_name="Admin",
                password_hash=hash_password(ADMIN_PASSWORD),
                is_admin=True,
                is_active=True,
            )
        )
        session.commit()
    finally:
        session.close()
    return login_staff(client, ADMIN_EMAIL, ADMIN_PASSWORD)


def _create(client: TestClient, token: str, **overrides) -> dict:
    payload = {
        "key": "shop",
        "name": "Shop",
        "greeting": "Hello",
        "privacy_url": "https://shop.example/privacy",
        "origins": ["https://shop.example"],
    }
    payload.update(overrides)
    return client.post("/api/sites", headers=_auth(token), json=payload)


def test_new_site_has_an_empty_profile(client: TestClient) -> None:
    token = _login_admin(client)
    body = _create(client, token).json()
    assert body["assistant_scope"] == ""
    assert body["handoff_topics"] == []
    assert body["primary_language"] == "en"


def test_create_site_stores_profile(client: TestClient) -> None:
    token = _login_admin(client)
    created = _create(
        client,
        token,
        assistant_scope=f"  {SCOPE}  ",
        handoff_topics=["  refunds   over 500 ", "Refunds over 500", "", "legal threats"],
        primary_language="pt-BR",
    )
    assert created.status_code == 201
    body = created.json()
    assert body["assistant_scope"] == SCOPE
    assert body["handoff_topics"] == ["refunds over 500", "legal threats"]
    assert body["primary_language"] == "pt-BR"


def test_patch_updates_and_clears_profile_independently(client: TestClient) -> None:
    token = _login_admin(client)
    site_id = _create(client, token, assistant_scope=SCOPE, handoff_topics=["a topic"]).json()["id"]
    patched = client.patch(
        f"/api/sites/{site_id}", headers=_auth(token), json={"primary_language": "de"}
    )
    assert patched.status_code == 200
    assert patched.json()["assistant_scope"] == SCOPE
    assert patched.json()["handoff_topics"] == ["a topic"]
    assert patched.json()["primary_language"] == "de"
    cleared = client.patch(
        f"/api/sites/{site_id}",
        headers=_auth(token),
        json={"assistant_scope": "", "handoff_topics": []},
    )
    assert cleared.status_code == 200
    assert cleared.json()["assistant_scope"] == ""
    assert cleared.json()["handoff_topics"] == []
    listed = client.get("/api/sites", headers=_auth(token)).json()["items"][0]
    assert listed["primary_language"] == "de"


@pytest.mark.parametrize(
    "payload",
    [
        {"assistant_scope": "x" * 2001},
        {"handoff_topics": [f"topic {n}" for n in range(21)]},
        {"handoff_topics": ["x" * 201]},
        {"primary_language": "english"},
        {"primary_language": "e"},
        {"primary_language": "en_US"},
        {"primary_language": ""},
    ],
)
def test_invalid_profile_is_rejected(client: TestClient, payload: dict) -> None:
    token = _login_admin(client)
    site_id = _create(client, token).json()["id"]
    response = client.patch(f"/api/sites/{site_id}", headers=_auth(token), json=payload)
    assert response.status_code == 422
    assert _create(client, token, key="other", **payload).status_code == 422


def test_scope_at_the_limit_is_accepted(client: TestClient) -> None:
    token = _login_admin(client)
    created = _create(client, token, assistant_scope="x" * 2000)
    assert created.status_code == 201


def _insert_conversation(site_id: uuid.UUID, intent: str | None) -> None:
    session = next(sync_session())
    try:
        visitor_id = uuid.uuid4()
        session.execute(
            text("INSERT INTO visitors (id, site_id, resume_token_hash) VALUES (:v, :s, :h)"),
            {"v": visitor_id, "s": site_id, "h": uuid.uuid4().hex},
        )
        session.execute(
            text(
                "INSERT INTO conversations (id, site_id, visitor_id, state, intent) "
                "VALUES (:c, :s, :v, 'bot', :i)"
            ),
            {"c": uuid.uuid4(), "s": site_id, "v": visitor_id, "i": intent},
        )
        session.commit()
    finally:
        session.close()


def test_conversation_intent_is_free_text(migrated_db) -> None:
    site_id = insert_site("intent-site", "Intent", uuid.uuid4().hex)
    _insert_conversation(site_id, "shipping_times")
    _insert_conversation(site_id, None)
    with pytest.raises(IntegrityError):
        _insert_conversation(site_id, "x" * 65)


def test_profile_migration_round_trip_keeps_sites_and_intents(migrated_db) -> None:
    site_id = insert_site("round-trip", "Round trip", uuid.uuid4().hex)
    _insert_conversation(site_id, "custom_topic")
    config = Config("alembic.ini")
    engine = create_engine(TEST_DATABASE_URL.replace("+asyncpg", "+psycopg"))
    command.downgrade(config, "56workspacesearch")
    try:
        with engine.connect() as connection:
            columns = {
                row[0]
                for row in connection.execute(
                    text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_name = 'sites'"
                    )
                )
            }
            assert not {"assistant_scope", "handoff_topics", "primary_language"} & columns
            assert connection.execute(
                text("SELECT intent FROM conversations WHERE intent IS NOT NULL")
            ).scalars().all() == ["other"]
        command.upgrade(config, "head")
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT assistant_scope, handoff_topics, primary_language "
                    "FROM sites WHERE id = :id"
                ),
                {"id": site_id},
            ).one()
            assert tuple(row) == ("", [], "en")
    finally:
        command.upgrade(config, "head")
        engine.dispose()
