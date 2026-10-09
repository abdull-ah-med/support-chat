import csv
import io
import json
from uuid import UUID

from fastapi.testclient import TestClient

from app.services.canned_import import ExistingCanned, plan_import
from tests.ws_helpers import (
    ALEX_PASSWORD,
    EASY_PUBLIC_KEY,
    HOST_ORIGIN,
    insert_site,
    insert_staff,
    login_staff,
)

ADMIN_EMAIL = "admin@example.local"

SHOP = UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
BLOG = UUID("dddddddd-dddd-4ddd-8ddd-dddddddddddd")
EXISTING = UUID("cccccccc-cccc-4ccc-8ccc-cccccccccccc")

HOURS_BODY = "We are open from nine to five on weekdays."
UPDATED_HOURS = "We are open from eight to six on weekdays."
RETURNS_BODY = "Returns are accepted within 30 days."
GREETING_BODY = "Thank you for contacting us. How may I assist you today?"
FILL_IN_BODY = "Your request was received on [DATE/TIME] under reference {Reference}."
NAMED_BODY = "Hi %customer-name%, we are open from nine to five on weekdays."


def _csv_row(
    livechat_id: int,
    body: str,
    group: int,
    tags: str = '["hours"]',
) -> dict[str, str]:
    return {"id": str(livechat_id), "text": body, "tags": tags, "group": str(group)}


def _existing(
    *,
    livechat_id: int | None = 200,
    body: str = HOURS_BODY,
    shortcut: str = "hours",
    site_id: UUID | None = SHOP,
    aliases: tuple[str, ...] = (),
    bot_eligible: bool = True,
    suggestion_event: str | None = None,
    hands_off: bool = False,
) -> ExistingCanned:
    return ExistingCanned(
        id=EXISTING,
        site_id=site_id,
        shortcut=shortcut,
        aliases=aliases,
        external_id=livechat_id,
        body=body,
        bot_eligible=bot_eligible,
        suggestion_event=suggestion_event,
        hands_off=hands_off,
    )


def test_groups_other_than_the_default_are_unmapped_until_staff_choose_a_site() -> None:
    """Catches a group being guessed onto a website from its number or name."""
    planned = plan_import([_csv_row(501, RETURNS_BODY, 5, '["returns"]')], existing=[])
    assert len(planned) == 1
    row = planned[0]
    assert row.action == "unmapped"
    assert row.group == 5
    assert row.site_id is None
    assert row.excerpt == RETURNS_BODY
    assert row.body == RETURNS_BODY
    assert row.shortcut == "returns"
    assert row.reason is None


def test_default_group_imports_as_general() -> None:
    """Catches shared responses needing a website choice."""
    planned = plan_import([_csv_row(1, HOURS_BODY, 0)], existing=[])
    assert planned[0].action == "create"
    assert planned[0].site_id is None


def test_a_group_can_be_assigned_to_any_website_and_none_means_general() -> None:
    planned = plan_import(
        [_csv_row(501, RETURNS_BODY, 5, '["returns"]'), _csv_row(502, HOURS_BODY, 6)],
        existing=[],
        remap_groups={5: BLOG, 6: None},
    )
    assert [(row.action, row.site_id) for row in planned] == [("create", BLOG), ("create", None)]


def test_a_reimported_response_keeps_the_website_it_was_imported_to() -> None:
    """Catches a re-import asking staff to pick the website again for known responses."""
    planned = plan_import(
        [_csv_row(200, UPDATED_HOURS, 6), _csv_row(300, RETURNS_BODY, 6, '["returns"]')],
        existing=[_existing()],
    )
    assert planned[0].action == "update"
    assert planned[0].site_id == SHOP
    assert planned[1].action == "unmapped"


def test_remapping_an_imported_response_to_another_website_is_skipped() -> None:
    planned = plan_import(
        [_csv_row(200, HOURS_BODY, 6)], existing=[_existing()], remap_groups={6: BLOG}
    )
    assert planned[0].action == "skip"
    assert planned[0].reason == "This LiveChat id is already imported in a different scope."


def test_rows_with_agent_fill_in_fields_are_not_bot_eligible() -> None:
    """Catches '{Reference}' and '[DATE/TIME]' scripts being enabled for the bot on import."""
    planned = plan_import(
        [
            _csv_row(49, FILL_IN_BODY, 6, '["received"]'),
            _csv_row(77, NAMED_BODY, 6, '["named"]'),
        ],
        existing=[],
        remap_groups={6: SHOP},
    )
    by_id = {row.livechat_id: row for row in planned}
    assert by_id[49].bot_eligible is False
    assert by_id[49].disable_reason == "Has fill-in fields — not available to the bot."
    assert by_id[77].bot_eligible is True
    assert by_id[77].disable_reason is None


def test_event_tagged_responses_are_not_bot_eligible() -> None:
    planned = plan_import(
        [_csv_row(97, GREETING_BODY, 0, '["_onstartchat_"]')],
        existing=[],
    )
    assert planned[0].suggestion_event == "start_chat"
    assert planned[0].bot_eligible is False
    assert planned[0].disable_reason == "Greeting or idle prompt — not available to the bot."


def test_no_tag_or_wording_is_treated_as_special_by_the_importer() -> None:
    """Catches built-in tag lists deciding which responses the bot may use or hand off."""
    planned = plan_import(
        [
            _csv_row(1, "Use code SAVE10 at checkout.", 0, '["20", "greeting", "verified"]'),
            _csv_row(2, "I will document this and route it.", 0, '["protected"]'),
        ],
        existing=[],
    )
    assert [row.bot_eligible for row in planned] == [True, True]
    assert [row.hands_off for row in planned] == [False, False]


def test_reimport_preserves_staff_choices_about_the_bot() -> None:
    """Catches an import re-enabling a response staff switched off, or clearing hand-off."""
    planned = plan_import(
        [_csv_row(200, UPDATED_HOURS, 6)],
        existing=[_existing(bot_eligible=False, hands_off=True)],
    )
    assert planned[0].action == "update"
    assert planned[0].bot_eligible is False
    assert planned[0].hands_off is True


def test_commit_refuses_unmapped_groups(client: TestClient) -> None:
    """Catches a group being silently dropped when staff omit a mapping."""
    insert_site("shop", "Shop", EASY_PUBLIC_KEY, [HOST_ORIGIN])
    insert_staff(ADMIN_EMAIL, "Admin", ALEX_PASSWORD, is_admin=True)
    token = login_staff(client, ADMIN_EMAIL, ALEX_PASSWORD)
    payload = _csv_bytes([_csv_row(200, HOURS_BODY, 0), _csv_row(501, RETURNS_BODY, 5)])
    commit = client.post(
        "/api/canned-replies/import",
        headers=_auth(token),
        files={"file": ("canned.csv", payload, "text/csv")},
        data={"decisions": json.dumps({"discard_ids": [], "remap_groups": {}})},
    )
    assert commit.status_code == 422
    assert commit.json()["detail"] == (
        "Choose a website for each LiveChat group, or discard those responses."
    )
    library = client.get("/api/canned-replies/library", headers=_auth(token))
    assert library.json()["items"] == []


def test_same_livechat_response_with_no_edits_is_unchanged() -> None:
    """Catches a re-import treating identical LiveChat copy as an update."""
    planned = plan_import([_csv_row(200, HOURS_BODY, 6)], existing=[_existing()])
    assert planned[0].action == "unchanged"
    assert planned[0].existing_id == EXISTING
    assert planned[0].excerpt == HOURS_BODY


def test_same_livechat_id_with_changed_copy_is_an_update() -> None:
    """Catches a changed LiveChat body being skipped or created as a new shortcut."""
    planned = plan_import([_csv_row(200, UPDATED_HOURS, 6)], existing=[_existing()])
    assert planned[0].action == "update"
    assert planned[0].existing_id == EXISTING
    assert planned[0].body == UPDATED_HOURS
    assert planned[0].excerpt == UPDATED_HOURS


def test_matching_shortcut_in_the_same_scope_is_an_update_even_without_livechat_id() -> None:
    """Catches a library row without external_id being duplicated by shortcut."""
    planned = plan_import(
        [_csv_row(200, UPDATED_HOURS, 6)],
        existing=[_existing(livechat_id=None)],
        remap_groups={6: SHOP},
    )
    assert planned[0].action == "update"
    assert planned[0].existing_id == EXISTING
    assert planned[0].body == UPDATED_HOURS


def test_discarded_livechat_ids_are_skipped() -> None:
    """Catches discard decisions still writing an update."""
    planned = plan_import(
        [_csv_row(200, UPDATED_HOURS, 6)],
        existing=[_existing()],
        discard_ids={200},
    )
    assert planned[0].action == "skip"
    assert planned[0].reason == "Discarded."
    assert planned[0].excerpt == UPDATED_HOURS


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _csv_bytes(rows: list[dict[str, str]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=["id", "text", "tags", "group"])
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def test_preview_keeps_copy_and_commit_places_each_group_where_staff_chose(
    client: TestClient,
) -> None:
    """Catches groups vanishing or landing on a website staff did not choose."""
    shop_id = insert_site("shop", "Shop", EASY_PUBLIC_KEY, [HOST_ORIGIN])
    blog_id = insert_site("blog", "Blog", "f" * 64, [HOST_ORIGIN])
    insert_staff(ADMIN_EMAIL, "Admin", ALEX_PASSWORD, is_admin=True)
    token = login_staff(client, ADMIN_EMAIL, ALEX_PASSWORD)
    payload = _csv_bytes(
        [
            _csv_row(100, GREETING_BODY, 0, '["welcome"]'),
            _csv_row(200, HOURS_BODY, 6),
            _csv_row(410, RETURNS_BODY, 4, '["returns"]'),
        ]
    )

    preview = client.post(
        "/api/canned-replies/import/preview",
        headers=_auth(token),
        files={"file": ("canned.csv", payload, "text/csv")},
    )
    assert preview.status_code == 200
    rows = {row["livechat_id"]: row for row in preview.json()["rows"]}
    assert rows[100]["action"] == "create"
    assert rows[100]["site_id"] is None
    assert rows[200]["action"] == "unmapped"
    assert rows[200]["excerpt"] == HOURS_BODY
    assert rows[410]["action"] == "unmapped"
    assert rows[410]["excerpt"] == RETURNS_BODY
    assert "group_name" not in rows[200]
    assert "livechat_website" not in rows[200]

    commit = client.post(
        "/api/canned-replies/import",
        headers=_auth(token),
        files={"file": ("canned.csv", payload, "text/csv")},
        data={
            "decisions": json.dumps(
                {"discard_ids": [], "remap_groups": {"4": str(blog_id), "6": str(shop_id)}}
            )
        },
    )
    assert commit.status_code == 200
    assert commit.json() == {"created": 3, "updated": 0, "skipped": 0}
    library = client.get("/api/canned-replies/library", headers=_auth(token))
    items = {item["shortcut"]: item for item in library.json()["items"]}
    assert items["welcome"]["site_id"] is None
    assert items["hours"]["site_id"] == str(shop_id)
    assert items["hours"]["body"] == HOURS_BODY
    assert items["returns"]["site_id"] == str(blog_id)
    assert items["returns"]["body"] == RETURNS_BODY

    again = client.post(
        "/api/canned-replies/import/preview",
        headers=_auth(token),
        files={"file": ("canned.csv", payload, "text/csv")},
    )
    assert {row["action"] for row in again.json()["rows"]} == {"unchanged"}


def test_commit_can_leave_an_existing_response_unchanged_when_staff_discard_the_update(
    client: TestClient,
) -> None:
    """Catches a re-import overwriting library copy the specialist chose to keep."""
    shop_id = insert_site("shop", "Shop", EASY_PUBLIC_KEY, [HOST_ORIGIN])
    insert_staff(ADMIN_EMAIL, "Admin", ALEX_PASSWORD, is_admin=True)
    token = login_staff(client, ADMIN_EMAIL, ALEX_PASSWORD)
    first = _csv_bytes([_csv_row(200, HOURS_BODY, 6)])
    imported = client.post(
        "/api/canned-replies/import",
        headers=_auth(token),
        files={"file": ("canned.csv", first, "text/csv")},
        data={"decisions": json.dumps({"discard_ids": [], "remap_groups": {"6": str(shop_id)}})},
    )
    assert imported.status_code == 200
    assert imported.json()["created"] == 1

    changed = _csv_bytes([_csv_row(200, UPDATED_HOURS, 6)])
    preview = client.post(
        "/api/canned-replies/import/preview",
        headers=_auth(token),
        files={"file": ("canned.csv", changed, "text/csv")},
    )
    assert preview.status_code == 200
    assert preview.json()["rows"][0]["action"] == "update"
    assert preview.json()["rows"][0]["excerpt"] == UPDATED_HOURS

    discarded = client.post(
        "/api/canned-replies/import",
        headers=_auth(token),
        files={"file": ("canned.csv", changed, "text/csv")},
        data={"decisions": json.dumps({"discard_ids": [200], "remap_groups": {}})},
    )
    assert discarded.status_code == 200
    assert discarded.json() == {"created": 0, "updated": 0, "skipped": 1}
    library = client.get("/api/canned-replies/library", headers=_auth(token))
    assert library.json()["items"][0]["body"] == HOURS_BODY
    assert library.json()["items"][0]["site_id"] == str(shop_id)


def test_fill_in_response_preview_names_why_it_is_disabled(client: TestClient) -> None:
    """Catches fill-in scripts staying bot-eligible after commit."""
    insert_staff(ADMIN_EMAIL, "Admin", ALEX_PASSWORD, is_admin=True)
    token = login_staff(client, ADMIN_EMAIL, ALEX_PASSWORD)
    payload = _csv_bytes([_csv_row(310, FILL_IN_BODY, 0, '["received"]')])
    preview = client.post(
        "/api/canned-replies/import/preview",
        headers=_auth(token),
        files={"file": ("canned.csv", payload, "text/csv")},
    )
    assert preview.status_code == 200
    row = preview.json()["rows"][0]
    assert row["action"] == "create"
    assert row["bot_eligible"] is False
    assert row["disable_reason"] == "Has fill-in fields — not available to the bot."
    commit = client.post(
        "/api/canned-replies/import",
        headers=_auth(token),
        files={"file": ("canned.csv", payload, "text/csv")},
    )
    assert commit.status_code == 200
    item = client.get("/api/canned-replies/library", headers=_auth(token)).json()["items"][0]
    assert item["shortcut"] == "received"
    assert item["bot_eligible"] is False
