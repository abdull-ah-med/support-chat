from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from uuid import UUID

BODY_MAX = 20_000


class CannedCsvError(Exception):
    pass


_SHORTCUT = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
_NON_SLUG = re.compile(r"[^a-z0-9_-]+")
_REPEAT_HYPHEN = re.compile(r"-{2,}")

EVENT_TAGS = {
    "_onstartchat_": "start_chat",
    "_onidle_": "idle",
    "_ongoodrate_": "good_rate",
    "_onbadrate_": "bad_rate",
    "_ontransferchat_": "transfer",
}

# LiveChat's built-in default group. It holds responses shared by every website; every
# other group is mapped to a website by the staff member running the import.
GENERAL_GROUP = 0
# Agent fill-ins such as {DOTAgency} or [DATE/TIME]. The four official %variables% are not these.
_UNFILLED_FIELD = re.compile(r"\{[^{}\s][^{}]*\}|\[[A-Z][A-Z0-9 /#_-]*\]")
FILL_IN_REASON = "Has fill-in fields — not available to the bot."

REQUIRED_COLUMNS = ("id", "text", "tags", "group")


@dataclass(frozen=True)
class ExistingCanned:
    id: UUID
    site_id: UUID | None
    shortcut: str
    aliases: tuple[str, ...]
    external_id: int | None
    body: str
    bot_eligible: bool
    suggestion_event: str | None
    hands_off: bool = False


@dataclass(frozen=True)
class ImportPlanRow:
    action: str
    reason: str | None
    livechat_id: int | None
    group: int | None
    site_id: UUID | None
    existing_id: UUID | None
    shortcut: str
    aliases: tuple[str, ...]
    body: str
    suggestion_event: str | None
    bot_eligible: bool
    disable_reason: str | None
    excerpt: str
    hands_off: bool = False


def slugify_tag(raw: str) -> str | None:
    lowered = raw.strip().lower()
    if lowered in EVENT_TAGS:
        return None
    slug = _REPEAT_HYPHEN.sub("-", _NON_SLUG.sub("-", lowered)).strip("-_")
    slug = slug[:40].strip("-_")
    if not slug or not _SHORTCUT.fullmatch(slug):
        return None
    return slug


def unique_shortcut(base: str, taken: set[str]) -> str | None:
    if base not in taken:
        return base
    for index in range(2, 100):
        suffix = f"-{index}"
        candidate = f"{base[: 40 - len(suffix)]}{suffix}"
        if candidate not in taken:
            return candidate
    return None


def has_unfilled_fields(body: str) -> bool:
    return _UNFILLED_FIELD.search(body) is not None


def bot_block_reason(body: str) -> str | None:
    """Why the bot cannot use this wording even when the row is switched on."""
    return FILL_IN_REASON if has_unfilled_fields(body) else None


def bot_eligible_for(suggestion_event: str | None, body: str) -> bool:
    return disable_reason_for(suggestion_event, body) is None


def disable_reason_for(suggestion_event: str | None, body: str) -> str | None:
    if suggestion_event is not None:
        return "Greeting or idle prompt — not available to the bot."
    return bot_block_reason(body)


def parse_livechat_csv(raw: bytes) -> list[dict[str, str]]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CannedCsvError from exc
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise CannedCsvError
    columns = {name.strip().lstrip("\ufeff") for name in reader.fieldnames if name}
    if any(column not in columns for column in REQUIRED_COLUMNS):
        raise CannedCsvError
    return [{(key or "").strip(): value or "" for key, value in row.items()} for row in reader]


def plan_import(
    rows: list[dict[str, str]],
    *,
    existing: list[ExistingCanned],
    remap_groups: dict[int, UUID | None] | None = None,
    discard_ids: set[int] | None = None,
) -> list[ImportPlanRow]:
    """Plan an import. A LiveChat group lands on the website the staff member chose for it
    (``remap_groups``; ``None`` means General). Without a choice, a group resolves only from
    LiveChat's default group or from responses of that id that were imported before."""
    by_external = {row.external_id: row for row in existing if row.external_id is not None}
    by_token: dict[tuple[UUID | None, str], ExistingCanned] = {}
    taken: dict[UUID | None, set[str]] = {}
    for row in existing:
        taken.setdefault(row.site_id, set()).update((row.shortcut, *row.aliases))
        by_token[(row.site_id, row.shortcut)] = row
        for alias in row.aliases:
            by_token.setdefault((row.site_id, alias), row)
    remaps = remap_groups or {}
    discarded = discard_ids or set()
    return [
        _plan_row(
            raw,
            by_external=by_external,
            by_token=by_token,
            taken=taken,
            remap_groups=remaps,
            discard_ids=discarded,
        )
        for raw in rows
    ]


def _plan_row(
    raw: dict[str, str],
    *,
    by_external: dict[int, ExistingCanned],
    by_token: dict[tuple[UUID | None, str], ExistingCanned],
    taken: dict[UUID | None, set[str]],
    remap_groups: dict[int, UUID | None],
    discard_ids: set[int],
) -> ImportPlanRow:
    livechat_id = _parse_int(raw.get("id", ""))
    group = _parse_int(raw.get("group", ""))
    body = (raw.get("text") or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if livechat_id is None:
        return _skip(None, group, "Row is missing a LiveChat id.")
    if group is None:
        return _skip(livechat_id, None, "Row is missing a LiveChat group.")
    parsed = _parse_body_and_tags(livechat_id, group, body, raw.get("tags") or "[]")
    if isinstance(parsed, ImportPlanRow):
        return parsed
    slugs, suggestion_event = parsed
    excerpt = _excerpt(body)
    if livechat_id in discard_ids:
        return _skip(livechat_id, group, "Discarded.", body=body, excerpt=excerpt)
    site_id, mapped = _resolve_site(group, livechat_id, by_external, remap_groups)
    if not slugs:
        slugs = [f"lc-{livechat_id}"[:40]]
    if not mapped:
        return _unmapped_row(livechat_id, group, slugs, suggestion_event, body, excerpt)
    return _mapped_row(
        livechat_id=livechat_id,
        group=group,
        site_id=site_id,
        slugs=slugs,
        suggestion_event=suggestion_event,
        body=body,
        excerpt=excerpt,
        by_external=by_external,
        by_token=by_token,
        taken=taken,
    )


def _parse_body_and_tags(
    livechat_id: int,
    group: int,
    body: str,
    raw_tags: str,
) -> tuple[list[str], str | None] | ImportPlanRow:
    if not body:
        return _skip(livechat_id, group, "Message is empty.")
    if len(body) > BODY_MAX:
        return _skip(
            livechat_id,
            group,
            f"Message is longer than {BODY_MAX:,} characters.",
            body=body[:120],
            excerpt=_excerpt(body),
        )
    try:
        tags = json.loads(raw_tags)
    except json.JSONDecodeError:
        return _skip(
            livechat_id,
            group,
            "Tags are not valid JSON.",
            body=body,
            excerpt=_excerpt(body),
        )
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        return _skip(
            livechat_id,
            group,
            "Tags must be a JSON array of strings.",
            body=body,
            excerpt=_excerpt(body),
        )
    suggestion_event = None
    slugs: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        event = EVENT_TAGS.get(tag.strip().lower())
        if event is not None:
            if suggestion_event is None:
                suggestion_event = event
            continue
        slug = slugify_tag(tag)
        if slug is None or slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    return slugs, suggestion_event


def _resolve_site(
    group: int,
    livechat_id: int,
    by_external: dict[int, ExistingCanned],
    remap_groups: dict[int, UUID | None],
) -> tuple[UUID | None, bool]:
    """Return (site_id, mapped). Nothing here knows which websites exist or what they are called."""
    if group in remap_groups:
        return remap_groups[group], True
    if group == GENERAL_GROUP:
        return None, True
    previous = by_external.get(livechat_id)
    if previous is not None:
        return previous.site_id, True
    return None, False


def _unmapped_row(
    livechat_id: int,
    group: int,
    slugs: list[str],
    suggestion_event: str | None,
    body: str,
    excerpt: str,
) -> ImportPlanRow:
    return ImportPlanRow(
        action="unmapped",
        reason=None,
        livechat_id=livechat_id,
        group=group,
        site_id=None,
        existing_id=None,
        shortcut=slugs[0],
        aliases=tuple(slugs[1:]),
        body=body,
        suggestion_event=suggestion_event,
        bot_eligible=bot_eligible_for(suggestion_event, body),
        disable_reason=disable_reason_for(suggestion_event, body),
        excerpt=excerpt,
    )


def _mapped_row(
    *,
    livechat_id: int,
    group: int,
    site_id: UUID | None,
    slugs: list[str],
    suggestion_event: str | None,
    body: str,
    excerpt: str,
    by_external: dict[int, ExistingCanned],
    by_token: dict[tuple[UUID | None, str], ExistingCanned],
    taken: dict[UUID | None, set[str]],
) -> ImportPlanRow:
    existing_row = _match_existing(livechat_id, site_id, slugs[0], by_external, by_token)
    if existing_row is not None and existing_row.site_id != site_id:
        return _skip(
            livechat_id,
            group,
            "This LiveChat id is already imported in a different scope.",
            body=body,
            excerpt=excerpt,
        )
    scope_taken = taken.setdefault(site_id, set())
    reserved = set(scope_taken)
    if existing_row is not None:
        reserved.difference_update((existing_row.shortcut, *existing_row.aliases))
    shortcut = unique_shortcut(slugs[0], reserved)
    if shortcut is None:
        return _skip(
            livechat_id,
            group,
            "Could not build a valid shortcut from the tags.",
            body=body,
            excerpt=excerpt,
        )
    aliases = [slug for slug in slugs[1:] if slug != shortcut and slug not in reserved]
    reserved.update((shortcut, *aliases))
    scope_taken.clear()
    scope_taken.update(reserved)
    # Which responses the bot may use, and which hand the visitor to a person, are staff
    # decisions made per response. An import only withholds wording the bot cannot send as
    # written, and never overwrites a choice staff already made.
    eligible = bot_eligible_for(suggestion_event, body)
    hands_off = False
    if existing_row is not None:
        eligible = eligible and existing_row.bot_eligible
        hands_off = existing_row.hands_off
    if existing_row is None:
        action = "create"
    else:
        action = _existing_action(
            existing_row,
            livechat_id,
            site_id,
            shortcut,
            aliases,
            body,
            eligible,
            suggestion_event,
            hands_off,
        )
    return ImportPlanRow(
        action=action,
        reason="Already in the library." if action == "unchanged" else None,
        livechat_id=livechat_id,
        group=group,
        site_id=site_id,
        existing_id=None if existing_row is None else existing_row.id,
        shortcut=shortcut,
        aliases=tuple(aliases),
        body=body,
        suggestion_event=suggestion_event,
        bot_eligible=eligible,
        disable_reason=disable_reason_for(suggestion_event, body),
        excerpt=excerpt,
        hands_off=hands_off,
    )


def _match_existing(
    livechat_id: int,
    site_id: UUID | None,
    shortcut: str,
    by_external: dict[int, ExistingCanned],
    by_token: dict[tuple[UUID | None, str], ExistingCanned],
) -> ExistingCanned | None:
    found = by_external.get(livechat_id)
    if found is not None:
        return found
    candidate = by_token.get((site_id, shortcut))
    if candidate is None:
        return None
    if candidate.external_id is not None and candidate.external_id != livechat_id:
        return None
    return candidate


def _existing_action(
    existing: ExistingCanned,
    livechat_id: int,
    site_id: UUID | None,
    shortcut: str,
    aliases: list[str],
    body: str,
    bot_eligible: bool,
    suggestion_event: str | None,
    hands_off: bool,
) -> str:
    same = (
        existing.external_id == livechat_id
        and existing.site_id == site_id
        and existing.shortcut == shortcut
        and set(existing.aliases) == set(aliases)
        and existing.body == body
        and existing.bot_eligible == bot_eligible
        and existing.suggestion_event == suggestion_event
        and existing.hands_off == hands_off
    )
    return "unchanged" if same else "update"


def _skip(
    livechat_id: int | None,
    group: int | None,
    reason: str,
    *,
    body: str = "",
    excerpt: str = "",
) -> ImportPlanRow:
    return ImportPlanRow(
        action="skip",
        reason=reason,
        livechat_id=livechat_id,
        group=group,
        site_id=None,
        existing_id=None,
        shortcut="",
        aliases=(),
        body=body,
        suggestion_event=None,
        bot_eligible=False,
        disable_reason=None,
        excerpt=excerpt,
    )


def _excerpt(body: str) -> str:
    return body if len(body) <= 120 else f"{body[:117]}..."


def _parse_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped.isdigit():
        return None
    return int(stripped)
