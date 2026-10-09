"""Retrieve bot-eligible canned replies and send a winning row verbatim."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

import structlog
from sqlalchemy import func, literal_column, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.canned_reply import CannedReply
from app.models.visitor import Visitor
from app.repositories.canned_reply_repo import CannedReplyRepository
from app.services.canned_import import has_unfilled_fields
from app.services.grounded_response_types import ProviderStatus, ResponseDecision, ResponseOutcome
from app.services.kb_embedder import Embedder, default_embedder, rrf_merge
from app.services.kb_hybrid import ChunkHit, HybridKbSearch, best_cosine_of
from app.services.kb_tokens import search_tokens, tokenize, tsquery_text
from app.services.route_decision import live_snapshots_for_site

log = structlog.get_logger("canned_bot")

FTS_LIMIT = 20
DENSE_LIMIT = 20
CANNED_COSINE_FLOOR = 0.55
CANNED_COSINE_GAP = 0.08
# A trigger phrase counts as said when every word of one alias is in the visitor's message
# and those words are at least this share of the message. A longer, more specific question
# that merely contains a phrase is left to meaning (embeddings) and the knowledge base.
PHRASE_COVERAGE_MIN = 0.6
# Staff wrote the trigger phrase, so a phrase match needs only a sanity floor on meaning.
PHRASE_COSINE_FLOOR = 0.25
MAX_QUERY_TOKENS = 24
HANDOFF_REASON = "individual_case"
_SPACE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class CannedHit:
    id: UUID
    shortcut: str
    body: str
    rrf: float
    cosine: float | None
    lexical: bool
    aliases: tuple[str, ...] = ()
    follow_up: bool = False
    hands_off: bool = False
    # Share of the visitor's words taken up by the best-matching alias phrase (0 if none).
    phrase_coverage: float = 0.0


def canned_embed_text(aliases: list[str] | tuple[str, ...], body: str) -> str:
    """What a reply answers: the phrases that trigger it, then its body. Never its slug."""
    phrases = " | ".join(alias for alias in aliases if alias)
    return f"{phrases}\n{body}" if phrases else body


def fill_canned_variables(
    body: str,
    *,
    customer_name: str = "",
    customer_email: str = "",
    agent_name: str = "",
    agent_email: str = "",
) -> str:
    filled = (
        body.replace("%customer-name%", (customer_name or "").strip())
        .replace("%customer-email%", (customer_email or "").strip())
        .replace("%agent-name%", (agent_name or "").strip())
        .replace("%agent-email%", (agent_email or "").strip())
    )
    filled = _SPACE.sub(" ", filled)
    return "\n".join(line.rstrip() for line in filled.splitlines()).strip()


def phrase_coverage(visitor_text: str, aliases: list[str] | tuple[str, ...]) -> float:
    """How completely the visitor's message is one of the reply's trigger phrases."""
    query = set(tokenize(visitor_text))
    if not query:
        return 0.0
    best = 0.0
    for alias in aliases:
        words = set(tokenize(alias))
        if words and words <= query:
            best = max(best, len(words) / len(query))
    return best


def pick_canned_winner(hits: list[CannedHit], *, best_kb_cosine: float | None) -> CannedHit | None:
    """Choose the one reply that clearly answers the message, or none.

    Evidence is what a reply answers (its trigger phrases and body), never its slug:
    a message that is one of its trigger phrases wins outright; otherwise the nearest
    by meaning wins only with a clear margin over the runner-up and over the knowledge base.
    """
    if not hits:
        return None
    said = sorted(
        (hit for hit in hits if hit.phrase_coverage >= PHRASE_COVERAGE_MIN),
        key=lambda hit: (-hit.phrase_coverage, -(hit.cosine or 0.0), str(hit.id)),
    )
    if said:
        top = said[0]
        tied = [hit for hit in said[1:] if hit.phrase_coverage >= top.phrase_coverage]
        sane = top.cosine is None or top.cosine >= PHRASE_COSINE_FLOOR
        if sane and not any(_close(top, other) for other in tied):
            return top
    scored = sorted(
        (hit for hit in hits if hit.cosine is not None),
        key=lambda hit: (-(hit.cosine or 0.0), str(hit.id)),
    )
    if not scored:
        return None
    best = scored[0]
    if best.cosine is None or best.cosine < CANNED_COSINE_FLOOR:
        return None
    if best_kb_cosine is not None and best.cosine <= best_kb_cosine:
        return None
    runner_up = scored[1].cosine if len(scored) > 1 else None
    if runner_up is not None and best.cosine - runner_up < CANNED_COSINE_GAP:
        return None
    return best


def _close(first: CannedHit, second: CannedHit) -> bool:
    """Two replies the message names equally well; only meaning can separate them."""
    if first.cosine is None or second.cosine is None:
        return True
    return abs(first.cosine - second.cosine) < CANNED_COSINE_GAP


async def refresh_canned_embedding(reply: CannedReply, embedder: Embedder) -> None:
    if not reply.enabled or not reply.bot_eligible:
        reply.embedding = None
        reply.embedder_id = None
        return
    vectors = await embedder.embed_documents(
        [canned_embed_text(list(reply.aliases or []), reply.body)]
    )
    reply.embedding = vectors[0]
    reply.embedder_id = embedder.embedder_id


async def embed_replies(replies: list[CannedReply], embedder: Embedder | None = None) -> None:
    if not replies:
        return
    worker = embedder if embedder is not None else default_embedder()
    pending = [reply for reply in replies if reply.enabled and reply.bot_eligible]
    for reply in replies:
        if reply not in pending:
            reply.embedding = None
            reply.embedder_id = None
    if not pending:
        return
    try:
        vectors = await worker.embed_documents(
            [canned_embed_text(list(reply.aliases or []), reply.body) for reply in pending]
        )
        for reply, vector in zip(pending, vectors, strict=True):
            reply.embedding = vector
            reply.embedder_id = worker.embedder_id
    except Exception as exc:
        # A vector left over from the old wording would mislead search, so clear it. The worker
        # re-embeds these rows on its next pass (refresh_stale_embeddings).
        for reply in pending:
            reply.embedding = None
            reply.embedder_id = None
        log.warning("canned_embed_failed", count=len(pending), error_class=type(exc).__name__)


async def refresh_stale_embeddings(session: AsyncSession, embedder: Embedder, limit: int) -> int:
    """Embed usable rows that have no vector or one from another model. Returns rows fixed."""
    rows = await CannedReplyRepository(session).list_needing_embedding(embedder.embedder_id, limit)
    if not rows:
        return 0
    await embed_replies(rows, embedder)
    await session.commit()
    return sum(1 for row in rows if row.embedder_id == embedder.embedder_id)


def _in_context(rows: list[CannedReply], previous_locator: str | None) -> list[CannedReply]:
    """Drop follow-ups unless the bot's last message was the script they follow."""
    shortcut_by_id = {row.id: row.shortcut for row in rows}

    def available(row: CannedReply) -> bool:
        if row.follows_id is None:
            return True
        parent = shortcut_by_id.get(row.follows_id)
        return parent is not None and previous_locator == f"#{parent}"

    return [row for row in rows if available(row)]


async def search_canned(
    session: AsyncSession,
    embedder: Embedder,
    site_id: UUID,
    visitor_text: str,
    previous_locator: str | None = None,
) -> list[CannedHit]:
    # Rows already imported, or switched on by staff, may still hold agent fill-in fields.
    usable = [
        row
        for row in await CannedReplyRepository(session).list_bot_eligible(site_id)
        if not has_unfilled_fields(row.body)
    ]
    rows = _in_context(usable, previous_locator)
    if not rows:
        return []
    by_id = {row.id: row for row in rows}
    eligible_ids = list(by_id)
    fts_ids = await _fts_ids(session, visitor_text, eligible_ids)
    query_vector = await _query_vector(embedder, visitor_text)
    dense_ids, cosine_by_id = await _dense_ids(
        session, eligible_ids, query_vector, embedder.embedder_id
    )
    ranked = rrf_merge(fts_ids, dense_ids)
    lexical_ids = set(fts_ids)
    hits: list[CannedHit] = []
    seen: set[UUID] = set()
    for reply_id, rrf in ranked:
        row = by_id.get(reply_id)
        if row is None:
            continue
        seen.add(reply_id)
        hits.append(_to_hit(row, rrf, cosine_by_id.get(reply_id), visitor_text, lexical_ids))
    # A message that is itself a trigger phrase counts even when neither arm ranked the row.
    for row in rows:
        if row.id in seen:
            continue
        if phrase_coverage(visitor_text, list(row.aliases or [])) >= PHRASE_COVERAGE_MIN:
            hits.append(_to_hit(row, 0.0, cosine_by_id.get(row.id), visitor_text, lexical_ids))
    return hits


async def canned_reply_decision(
    session: AsyncSession,
    embedder: Embedder,
    site_id: UUID,
    visitor_text: str,
    visitor: Visitor | None,
    previous_locator: str | None = None,
    *,
    kb_hits: Sequence[ChunkHit] | None = None,
) -> ResponseDecision | None:
    """A canned reply that clearly answers the message, else ``None``.

    ``kb_hits`` are the hits of a knowledge search already run for this message; passing
    them spares a second search when the reply must beat the knowledge base.
    """
    hits = await search_canned(session, embedder, site_id, visitor_text, previous_locator)
    if not hits:
        return None
    best_kb_cosine = await _best_kb_cosine(session, embedder, site_id, visitor_text, kb_hits)
    winner = pick_canned_winner(hits, best_kb_cosine=best_kb_cosine)
    if winner is None:
        return None
    log.info("canned_turn", canned_id=str(winner.id), shortcut=winner.shortcut)
    body = fill_canned_variables(
        winner.body,
        customer_name=visitor.name if visitor is not None else "",
        customer_email=visitor.email if visitor is not None else "",
    )
    return ResponseDecision(
        ResponseOutcome.SYNTHESIZED_ANSWER,
        "canned_reply",
        body,
        provider_status=ProviderStatus.NOT_USED,
        display_locator=f"#{winner.shortcut}",
        handoff_reason=HANDOFF_REASON if winner.hands_off else None,
    )


async def inspect_canned_search(
    session: AsyncSession,
    embedder: Embedder,
    site_id: UUID,
    visitor_text: str,
) -> dict[str, object]:
    hits = await search_canned(session, embedder, site_id, visitor_text)
    best_kb_cosine = None
    if hits:
        best_kb_cosine = await _best_kb_cosine(session, embedder, site_id, visitor_text, None)
    winner = pick_canned_winner(hits, best_kb_cosine=best_kb_cosine)
    return {
        "winner": None if winner is None else winner.shortcut,
        "best_kb_cosine": best_kb_cosine,
        "hits": [
            {
                "shortcut": hit.shortcut,
                "lexical": hit.lexical,
                "rrf": hit.rrf,
                "cosine": hit.cosine,
            }
            for hit in hits[:8]
        ],
    }


def _to_hit(
    row: CannedReply,
    rrf: float,
    cosine_score: float | None,
    visitor_text: str,
    lexical_ids: set[UUID],
) -> CannedHit:
    aliases = tuple(row.aliases or [])
    coverage = phrase_coverage(visitor_text, aliases)
    return CannedHit(
        id=row.id,
        shortcut=row.shortcut,
        body=row.body,
        rrf=rrf,
        cosine=cosine_score,
        aliases=aliases,
        lexical=row.id in lexical_ids or coverage > 0.0,
        follow_up=row.follows_id is not None,
        hands_off=row.hands_off,
        phrase_coverage=coverage,
    )


async def _query_vector(embedder: Embedder, visitor_text: str) -> list[float] | None:
    try:
        return await embedder.embed_query(visitor_text)
    except Exception:
        log.info("canned_embed_query_failed")
        return None


async def _fts_ids(
    session: AsyncSession, visitor_text: str, eligible_ids: list[UUID]
) -> list[UUID]:
    tokens = search_tokens(visitor_text)[:MAX_QUERY_TOKENS]
    if not tokens:
        return []
    simple = literal_column("'simple'::regconfig")
    any_term = func.to_tsquery(simple, tsquery_text(tokens, "|"))
    all_terms = func.to_tsquery(simple, tsquery_text(tokens, "&"))
    matches_all = CannedReply.search_document.op("@@")(all_terms)
    rank = func.ts_rank_cd(CannedReply.search_document, any_term, 1)
    result = await session.execute(
        select(CannedReply.id)
        .where(
            CannedReply.id.in_(eligible_ids),
            CannedReply.search_document.op("@@")(any_term),
        )
        .order_by(matches_all.desc(), rank.desc(), CannedReply.id)
        .limit(FTS_LIMIT)
    )
    return list(result.scalars().all())


async def _dense_ids(
    session: AsyncSession,
    eligible_ids: list[UUID],
    query_vector: list[float] | None,
    embedder_id: str,
) -> tuple[list[UUID], dict[UUID, float]]:
    if query_vector is None:
        return [], {}
    distance = CannedReply.embedding.cosine_distance(query_vector)
    result = await session.execute(
        select(CannedReply.id, distance)
        .where(
            CannedReply.id.in_(eligible_ids),
            CannedReply.embedding.is_not(None),
            # Vectors from another model live in a different space; their scores mean nothing here.
            CannedReply.embedder_id == embedder_id,
        )
        .order_by(distance, CannedReply.id)
        .limit(DENSE_LIMIT)
    )
    ids: list[UUID] = []
    cosine_by_id: dict[UUID, float] = {}
    for reply_id, dist in result.all():
        similarity = 1.0 - float(dist if dist is not None else 1.0)
        ids.append(reply_id)
        cosine_by_id[reply_id] = similarity
    return ids, cosine_by_id


async def _best_kb_cosine(
    session: AsyncSession,
    embedder: Embedder,
    site_id: UUID,
    visitor_text: str,
    kb_hits: Sequence[ChunkHit] | None,
) -> float | None:
    """Closeness of the best knowledge passage, from hits the caller already holds if any."""
    if kb_hits is not None:
        reported = getattr(kb_hits, "best_cosine", None)
        return reported if reported is not None else best_cosine_of(kb_hits)
    snapshots = await live_snapshots_for_site(session, site_id)
    if not snapshots:
        return None
    hits = await HybridKbSearch(session).search_with_deferred_embed(embedder, site_id, visitor_text)
    return hits.best_cosine
