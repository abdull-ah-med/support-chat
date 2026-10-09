"""Per-site bot evals: an independent case spec scored against the reply a real turn produced.

A dataset is one JSON file per site, `tests/evals/dataset/<site_key>.json`. Cases never name a
brand in this module; the wording, expected copy and substrings live in the dataset. Scoring
looks at the observable contract of a turn, not just its words:

- state / reason / outcome the conversation ended in,
- whether the reply carried citations and whether every cited chunk belongs to the site,
- whether the bot abstained when the case says the answer is not in the site's knowledge,
- latency per turn.

Runs use whatever responder and embedder the caller passes. Tests inject doubles; the
runner script uses the configured providers and refuses to start without them.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.kb_chunk import KbChunk
from app.models.message import Message
from app.models.message_citation import MessageCitation
from app.models.site import Site
from app.models.visitor import Visitor
from app.repositories.message_repo import MessageRepository
from app.services.conversation_service import ConversationService

DATASET_DIR = Path(__file__).resolve().parents[2] / "tests" / "evals" / "dataset"

# Outcomes that present knowledge as an answer. Anything else is the bot declining or asking.
ANSWER_OUTCOMES = frozenset({"exact_answer", "synthesized_answer", "partial_answer"})


def dataset_path_for(site_key: str, directory: Path | None = None) -> Path:
    """`<dir>/<site_key>.json`, refusing keys that could point outside the directory."""
    if not site_key or "/" in site_key or "\\" in site_key or site_key.startswith("."):
        raise ValueError(f"invalid site key: {site_key!r}")
    return (directory or DATASET_DIR) / f"{site_key}.json"


@dataclass(frozen=True)
class EvalCase:
    id: str
    family: str
    turns: tuple[str, ...]
    expected_state: str
    expected_reason: str | None = None
    expected_copy: str | None = None
    expected_outcome: str | None = None
    expected_locator: str | None = None
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    # True: the answer is not in this site's knowledge, so the bot must decline (no answer
    # outcome, no citations). False/None: no abstention requirement.
    expected_unknown: bool = False
    # True: the reply must cite site knowledge. False: it must not. None: not checked.
    expect_citations: bool | None = None
    # Latency budget for the final turn in milliseconds. None: not checked.
    max_latency_ms: int | None = None


@dataclass(frozen=True)
class ObservedReply:
    state: str
    role: str
    body: str
    system_reason: str | None = None
    source_title: str | None = None
    outcome: str | None = None
    display_locator: str | None = None
    citation_chunk_ids: tuple[UUID, ...] = ()
    # Cited chunks whose own site differs from the site under test, or that no longer exist.
    foreign_citation_ids: tuple[UUID, ...] = ()
    unresolved_citation_ids: tuple[UUID, ...] = ()
    turn_latencies_ms: tuple[int, ...] = ()

    @property
    def cited(self) -> bool:
        return bool(self.citation_chunk_ids)

    @property
    def abstained(self) -> bool:
        """The bot declined: it neither cited knowledge nor presented an answer."""
        return not self.cited and self.outcome not in ANSWER_OUTCOMES

    @property
    def latency_ms(self) -> int | None:
        return self.turn_latencies_ms[-1] if self.turn_latencies_ms else None


@dataclass(frozen=True)
class EvalVerdict:
    case_id: str
    passed: bool
    failures: tuple[str, ...]
    family: str = ""
    outcome: str | None = None
    reason: str | None = None
    cited: bool = False
    abstained: bool = False
    expected_unknown: bool = False
    foreign_citations: int = 0
    latency_ms: int | None = None


def _meaning_failures(case: EvalCase, observed: ObservedReply) -> list[str]:
    failures: list[str] = []
    if observed.state != case.expected_state:
        failures.append(f"state: expected {case.expected_state}, got {observed.state}")
    if case.expected_reason is not None and observed.system_reason != case.expected_reason:
        failures.append(f"reason: expected {case.expected_reason}, got {observed.system_reason}")
    if case.expected_outcome is not None and observed.outcome != case.expected_outcome:
        failures.append(f"outcome: expected {case.expected_outcome}, got {observed.outcome}")
    if case.expected_locator is not None and observed.display_locator != case.expected_locator:
        failures.append(
            f"locator: expected {case.expected_locator}, got {observed.display_locator}"
        )
    if case.expected_copy is not None and observed.body != case.expected_copy:
        failures.append("copy: expected canned visitor line")
    lowered = (observed.body or "").casefold()
    failures += [f"must_contain: {n}" for n in case.must_contain if n.casefold() not in lowered]
    failures += [f"must_not_contain: {n}" for n in case.must_not_contain if n.casefold() in lowered]
    return failures


def _grounding_failures(case: EvalCase, observed: ObservedReply) -> list[str]:
    failures: list[str] = []
    if case.expected_unknown and not observed.abstained:
        failures.append(
            "abstain: answer is not in the site's knowledge but the bot "
            f"{'cited it' if observed.cited else 'answered'} (outcome={observed.outcome})"
        )
    if case.expect_citations is True and not observed.cited:
        failures.append("citations: expected the reply to cite site knowledge")
    if case.expect_citations is False and observed.cited:
        failures.append("citations: expected no citations")
    # Cross-site citations are never acceptable, whatever the case asks for.
    if observed.foreign_citation_ids:
        failures.append(f"foreign_citation: {len(observed.foreign_citation_ids)} chunk(s)")
    if observed.unresolved_citation_ids:
        failures.append(f"unverifiable_citation: {len(observed.unresolved_citation_ids)} chunk(s)")
    return failures


def score_case(case: EvalCase, observed: ObservedReply) -> EvalVerdict:
    failures = _meaning_failures(case, observed) + _grounding_failures(case, observed)
    if (
        case.max_latency_ms is not None
        and observed.latency_ms is not None
        and observed.latency_ms > case.max_latency_ms
    ):
        failures.append(f"latency: {observed.latency_ms}ms over {case.max_latency_ms}ms")
    return EvalVerdict(
        case_id=case.id,
        passed=not failures,
        failures=tuple(failures),
        family=case.family,
        outcome=observed.outcome,
        reason=observed.system_reason,
        cited=observed.cited,
        abstained=observed.abstained,
        expected_unknown=case.expected_unknown,
        foreign_citations=len(observed.foreign_citation_ids),
        latency_ms=observed.latency_ms,
    )


def case_from_dict(raw: dict) -> EvalCase:
    return EvalCase(
        id=str(raw["id"]),
        family=str(raw["family"]),
        turns=tuple(str(item) for item in raw["turns"]),
        expected_state=str(raw["expected_state"]),
        expected_reason=raw.get("expected_reason"),
        expected_copy=raw.get("expected_copy"),
        expected_outcome=raw.get("expected_outcome"),
        expected_locator=raw.get("expected_locator"),
        must_contain=tuple(raw.get("must_contain") or ()),
        must_not_contain=tuple(raw.get("must_not_contain") or ()),
        expected_unknown=bool(raw.get("expected_unknown", False)),
        expect_citations=raw.get("expect_citations"),
        max_latency_ms=raw.get("max_latency_ms"),
    )


def load_dataset(
    site_key: str | None = None, path: Path | None = None
) -> tuple[str, tuple[EvalCase, ...]]:
    """Cases for a site. `path` overrides the default `<DATASET_DIR>/<site_key>.json`.

    The returned key is the one asked for; a dataset's own "site_key" field is advisory and a
    mismatch is an error, so a file for one site is never run against another by accident.
    """
    if path is None:
        if site_key is None:
            raise ValueError("pass a site key or a dataset path")
        path = dataset_path_for(site_key)
    payload = json.loads(path.read_text())
    declared = payload.get("site_key")
    resolved = site_key or declared or path.stem
    if declared is not None and site_key is not None and declared != site_key:
        raise ValueError(f"{path.name} is a dataset for site {declared!r}, not {site_key!r}")
    cases = tuple(case_from_dict(item) for item in payload["cases"])
    ids = [case.id for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path.name} has duplicate case ids")
    return str(resolved), cases


async def open_eval_conversation(session: AsyncSession, site: Site) -> tuple[Visitor, Conversation]:
    visitor = Visitor(
        site_id=site.id,
        resume_token_hash=uuid4().hex,
        name="Eval Visitor",
        email="eval@supportchat.local",
    )
    session.add(visitor)
    await session.flush()
    conversation = Conversation(site_id=site.id, visitor_id=visitor.id, state="bot")
    session.add(conversation)
    await session.flush()
    return visitor, conversation


def _visible_reply(messages: list[Message]) -> Message | None:
    for row in reversed(messages):
        if row.role in {"bot", "system"}:
            return row
    return None


async def _citation_sites(
    session: AsyncSession, site_id: UUID, chunk_ids: Iterable[UUID]
) -> tuple[tuple[UUID, ...], tuple[UUID, ...]]:
    """Split cited chunk ids into (foreign to this site, no longer present)."""
    wanted = list(dict.fromkeys(chunk_ids))
    if not wanted:
        return (), ()
    rows = await session.execute(select(KbChunk.id, KbChunk.site_id).where(KbChunk.id.in_(wanted)))
    owner = {chunk_id: chunk_site for chunk_id, chunk_site in rows.all()}
    foreign = tuple(cid for cid in wanted if cid in owner and owner[cid] != site_id)
    missing = tuple(cid for cid in wanted if cid not in owner)
    return foreign, missing


async def _cited_chunk_ids(session: AsyncSession, row: Message) -> tuple[UUID, ...]:
    ids: list[UUID] = list(row.source_chunk_ids or [])
    result = await session.execute(
        select(MessageCitation.chunk_id).where(
            MessageCitation.message_id == row.id, MessageCitation.chunk_id.is_not(None)
        )
    )
    ids.extend(result.scalars().all())
    return tuple(dict.fromkeys(ids))


async def run_case(
    session: AsyncSession,
    site: Site,
    case: EvalCase,
    *,
    parent_origin: str,
    responder=None,
    embedder=None,
) -> tuple[ObservedReply, EvalVerdict]:
    visitor, conversation = await open_eval_conversation(session, site)
    await session.commit()
    service = ConversationService(session, responder=responder, embedder=embedder)
    latencies: list[int] = []
    for body in case.turns:
        started = time.perf_counter()
        result = await service.visitor_message(
            conversation.id, visitor.id, parent_origin, uuid4(), body
        )
        if result.generation_id is not None:
            await service.run_bot_turn(conversation.id, result.generation_id)
        latencies.append(round((time.perf_counter() - started) * 1000))
    await session.refresh(conversation)
    rows = await MessageRepository(session).list_recent_roles(
        conversation.id, {"bot", "system", "visitor"}, 20
    )
    visible = _visible_reply(list(rows))
    if visible is None:
        observed = ObservedReply(
            state=conversation.state, role="", body="", turn_latencies_ms=tuple(latencies)
        )
    else:
        cited = await _cited_chunk_ids(session, visible)
        foreign, missing = await _citation_sites(session, site.id, cited)
        observed = ObservedReply(
            state=conversation.state,
            role=visible.role,
            body=visible.body,
            system_reason=visible.system_reason,
            source_title=visible.source_title,
            outcome=visible.response_outcome,
            display_locator=visible.display_locator,
            citation_chunk_ids=cited,
            foreign_citation_ids=foreign,
            unresolved_citation_ids=missing,
            turn_latencies_ms=tuple(latencies),
        )
    return observed, score_case(case, observed)


@dataclass
class FamilyStats:
    family: str
    total: int = 0
    passed: int = 0
    unknown_cases: int = 0
    abstained_correctly: int = 0
    cited: int = 0
    foreign_citations: int = 0
    latencies: list[int] = field(default_factory=list)

    @property
    def p50_ms(self) -> int | None:
        return round(median(self.latencies)) if self.latencies else None

    @property
    def p95_ms(self) -> int | None:
        if not self.latencies:
            return None
        ordered = sorted(self.latencies)
        return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def summarize(verdicts: Iterable[EvalVerdict]) -> list[FamilyStats]:
    """Per-family counts: pass rate, abstention accuracy, citation rate, latency percentiles."""
    by_family: dict[str, FamilyStats] = defaultdict(lambda: FamilyStats(family=""))
    for verdict in verdicts:
        stats = by_family[verdict.family]
        stats.family = verdict.family
        stats.total += 1
        stats.passed += int(verdict.passed)
        stats.cited += int(verdict.cited)
        stats.foreign_citations += verdict.foreign_citations
        if verdict.expected_unknown:
            stats.unknown_cases += 1
            stats.abstained_correctly += int(verdict.abstained)
        if verdict.latency_ms is not None:
            stats.latencies.append(verdict.latency_ms)
    return sorted(by_family.values(), key=lambda item: item.family)


def format_summary(stats: Iterable[FamilyStats]) -> str:
    header = (
        f"{'family':28} {'pass':>9} {'abstain':>9} {'cited':>6} {'foreign':>8} "
        f"{'p50ms':>7} {'p95ms':>7}"
    )
    lines = [header]
    for item in stats:
        abstain = f"{item.abstained_correctly}/{item.unknown_cases}" if item.unknown_cases else "-"
        p50 = "-" if item.p50_ms is None else str(item.p50_ms)
        p95 = "-" if item.p95_ms is None else str(item.p95_ms)
        lines.append(
            f"{item.family:28} {f'{item.passed}/{item.total}':>9} {abstain:>9} "
            f"{item.cited:>6} {item.foreign_citations:>8} {p50:>7} {p95:>7}"
        )
    return "\n".join(lines)
