import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from urllib.parse import urlparse
from uuid import UUID

import structlog
from sqlalchemy import func, literal_column, select, text
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.display_citations import visitor_citation_url
from app.db import session_maker
from app.llm.safety_markers import contains_injection_marker
from app.models.kb_chunk import KbChunk
from app.models.kb_page import KbPage
from app.models.kb_snapshot import KbSnapshot
from app.models.kb_source import KbSource
from app.services.bot_trace import record_trace
from app.services.faq_fastpath import is_marketing_cta
from app.services.kb_embedder import rrf_merge
from app.services.kb_tokens import fold_text, search_tokens, token_key, tsquery_text
from app.settings import get_settings

log = structlog.get_logger("kb_hybrid")

# A hit whose embedding is this far from the question is dropped unless a lexical arm
# vouches for it strongly. It is also the floor callers use to judge "nothing relevant".
COSINE_FLOOR = 0.35
FTS_LIMIT = 20
DENSE_LIMIT = 20
# The nearest-neighbour scan is limited by chunk columns only (site, enabled, live
# snapshot) so the HNSW index serves it; page and source switches are applied to this
# over-fetched window afterwards.
DENSE_OVERFETCH = 60
HNSW_EF_SEARCH = 100
KEEP = 8
OVERVIEW_TOPICAL = 3
EXACT_SUPPORT = 3
EXACT_CANDIDATES = 40
MAX_QUERY_TOKENS = 24
# An all-terms text match only vouches for a hit when the question has this many terms.
MIN_STRONG_TOKENS = 2
TRGM_STRONG = 0.50
_TRGM_EXPR = (
    "workspace_search_fold("
    "coalesce(canonical_question, '') || ' ' || "
    "kb_aliases_as_text(aliases) || ' ' || "
    "coalesce(heading, '') || ' ' || "
    "coalesce(topic_label, '')"
    ")"
)

_ann_iterative_scan: bool | None = None


@dataclass(frozen=True)
class ChunkHit:
    id: UUID
    page_id: UUID
    site_id: UUID
    heading: str
    body: str
    url: str
    title: str
    rrf: float
    cosine: float | None
    answer_verbatim: str = ""
    display_locator: str | None = None
    snapshot_id: UUID | None = None
    canonical_question: str | None = None
    legal_sensitive: bool = False
    aliases: tuple[str, ...] = ()
    topic_label: str | None = None
    risk_class: str = "general"
    answer_mode: str = "paraphrase_allowed"
    enabled: bool = True
    structured: bool = False


class KbHits(list[ChunkHit]):
    """Retrieved chunks plus the strongest relevance signal seen while finding them.

    ``best_cosine`` is the highest embedding cosine among every candidate the search
    scored, including candidates later dropped for being too far from the question; an
    exact match to a stored question reports 1.0. ``None`` means no embedding was
    available to judge (provider failure, or a corpus without vectors).
    """

    def __init__(self, hits: Sequence[ChunkHit] = (), *, best_cosine: float | None = None) -> None:
        super().__init__(hits)
        self.best_cosine = best_cosine


@dataclass(frozen=True)
class _Arm:
    scored: list[tuple[UUID, float]]
    strong: frozenset[UUID] = frozenset()

    @property
    def ids(self) -> list[UUID]:
        return [chunk_id for chunk_id, _score in self.scored]


_NO_ARM = _Arm([])
_SIMPLE = literal_column("'simple'::regconfig")


def _elapsed_ms(started_ns: int) -> int:
    return (time.perf_counter_ns() - started_ns) // 1_000_000


def best_cosine_of(hits: Sequence[ChunkHit]) -> float | None:
    scores = [hit.cosine for hit in hits if hit.cosine is not None]
    return max(scores) if scores else None


def _cap_overview(hits: list[ChunkHit], keep: int = KEEP) -> list[ChunkHit]:
    kept: list[ChunkHit] = []
    seen: set[str] = set()
    for hit in hits:
        key = token_key(hit.heading) or str(hit.id)
        if key in seen:
            continue
        seen.add(key)
        kept.append(hit)
        if len(kept) >= keep:
            break
    return kept


def _eligible(site_id: UUID):
    """Conditions every retrieval arm shares: this site, enabled, in a live snapshot."""
    return (
        KbChunk.site_id == site_id,
        KbChunk.enabled.is_(True),
        KbChunk.kind != "refusal",
        KbPage.enabled.is_(True),
        KbSource.enabled.is_(True),
        KbSnapshot.state == "live",
    )


def _eligible_join(query):
    return (
        query.join(KbPage, KbPage.id == KbChunk.page_id)
        .join(KbSnapshot, KbSnapshot.id == KbChunk.snapshot_id)
        .join(KbSource, KbSource.id == KbPage.source_id)
    )


async def _embed_query(embedder, site_id: UUID, text_value: str, timings: dict[str, int]):
    started = time.perf_counter_ns()
    vector: list[float] | None = None
    try:
        vector = await embedder.embed_query(text_value)
    except Exception:
        log.info("grounded_embed_failed", site_id=str(site_id))
    timings["embed"] = _elapsed_ms(started)
    return vector


class HybridKbSearch:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def search(
        self,
        site_id: UUID,
        visitor_text: str,
        query_vector: list[float] | None = None,
        *,
        overview: bool = False,
    ) -> KbHits:
        return await self._retrieve(
            site_id,
            visitor_text,
            embedder=None,
            query_vector=query_vector,
            overview=overview,
            timings={},
        )

    async def search_with_deferred_embed(
        self,
        embedder,
        site_id: UUID,
        visitor_text: str,
        stage_timings: dict[str, int] | None = None,
        *,
        overview: bool = False,
    ) -> KbHits:
        """Search with the query embedded while the lexical arms run.

        ``overview`` is the caller's judgement (not a keyword guess here) that the
        visitor wants a general picture of the site; the start pages then lead.
        """
        timings = stage_timings if stage_timings is not None else {}
        return await self._retrieve(
            site_id,
            visitor_text,
            embedder=embedder,
            query_vector=None,
            overview=overview,
            timings=timings,
        )

    async def _retrieve(
        self,
        site_id: UUID,
        visitor_text: str,
        *,
        embedder,
        query_vector: list[float] | None,
        overview: bool,
        timings: dict[str, int],
    ) -> KbHits:
        exact = await self._exact_match(site_id, visitor_text)
        if exact is not None:
            # A stored question matched word for word: the vector adds nothing, so the
            # embedder stays idle. Keep the lexical neighbours as supporting context.
            embedder = None
            query_vector = None
            timings.setdefault("embed", 0)
            timings.setdefault("dense_retrieve", 0)

        fts_arm, trgm_arm, query_vector = await self._lexical_and_embed(
            site_id, visitor_text, embedder, query_vector, timings
        )

        dense_ids: list[UUID] = []
        cosine_by_id: dict[UUID, float] = {}
        timings["dense_retrieve"] = 0
        if query_vector is not None:
            started = time.perf_counter_ns()
            dense_ids, cosine_by_id = await self._dense(site_id, query_vector)
            timings["dense_retrieve"] = _elapsed_ms(started)

        ranked = [
            item
            for item in rrf_merge(fts_arm.ids, trgm_arm.ids, dense_ids)
            if exact is None or item[0] != exact.id
        ]
        record_trace(
            "retrieval",
            fts_scores=fts_arm.scored,
            trigram_scores=trgm_arm.scored,
            dense_cosines=cosine_by_id,
            query_vector_available=query_vector is not None,
        )
        wanted = [chunk_id for chunk_id, _score in ranked]
        hits = await self._load(site_id, wanted)
        if query_vector is not None:
            missing = [hit.id for hit in hits if hit.id not in cosine_by_id]
            cosine_by_id.update(await self._cosines(site_id, missing, query_vector))
        elif exact is not None:
            # No query vector was needed; judge neighbours by closeness to the matched passage.
            cosine_by_id.update(
                await self._neighbour_cosines(site_id, exact.id, [hit.id for hit in hits])
            )
        strong = fts_arm.strong | trgm_arm.strong
        selected = self._select(ranked, hits, cosine_by_id, strong, supporting=exact is not None)

        if exact is not None:
            front = [replace(exact, cosine=1.0)]
            selected = [*front, *selected][:KEEP]
            best = 1.0
            mode = "exact"
        else:
            best = max(cosine_by_id.values()) if cosine_by_id else None
            mode = "hybrid"
        if overview:
            led = await self._lead_with_start_pages(site_id, query_vector, selected, exact)
            if led is not None:
                selected, mode = led, "overview"
        record_trace("retrieval", mode=mode, hits=selected)
        if selected:
            log.info(
                "grounded_retrieve",
                hit_ids=[str(hit.id) for hit in selected],
                exact_match=exact is not None,
                overview_merged=mode == "overview",
                dense_used=bool(dense_ids),
            )
        return KbHits(selected, best_cosine=best)

    async def _lexical_and_embed(
        self,
        site_id: UUID,
        visitor_text: str,
        embedder,
        query_vector: list[float] | None,
        timings: dict[str, int],
    ) -> tuple[_Arm, _Arm, list[float] | None]:
        """Run both lexical arms while the question is embedded; return the vector (or None)."""
        started = time.perf_counter_ns()
        embed_task = (
            asyncio.create_task(_embed_query(embedder, site_id, visitor_text, timings))
            if embedder is not None
            else None
        )
        try:
            async with session_maker()() as trgm_session:
                fts_arm, trgm_arm = await asyncio.gather(
                    self._fts(site_id, visitor_text),
                    HybridKbSearch(trgm_session)._trigram(site_id, visitor_text),
                )
            if embed_task is not None:
                query_vector = await embed_task
        finally:
            if embed_task is not None and not embed_task.done():
                embed_task.cancel()
        lexical_ms = _elapsed_ms(started)
        timings["lexical_retrieve"] = lexical_ms
        timings["trigram_retrieve"] = lexical_ms
        timings.setdefault("embed", 0)
        return fts_arm, trgm_arm, query_vector

    async def _lead_with_start_pages(
        self,
        site_id: UUID,
        query_vector: list[float] | None,
        selected: list[ChunkHit],
        exact: ChunkHit | None,
    ) -> list[ChunkHit] | None:
        starts = await self._start_pages(site_id, query_vector)
        if not starts:
            return None
        lead = [hit for hit in starts if exact is None or hit.id != exact.id]
        # Start pages introduce the site, but the best topical matches must still reach the
        # model: a broad question that names a subject would otherwise lose its own evidence.
        floor = get_settings().evidence_min_cosine
        topical = [hit for hit in selected if (hit.cosine or 0.0) >= floor][:OVERVIEW_TOPICAL]
        topical_ids = {hit.id for hit in topical}
        rest = [hit for hit in selected if hit.id not in topical_ids]
        return _cap_overview([*topical, *lead, *rest])

    def _select(
        self,
        ranked: list[tuple[UUID, float]],
        hits: list[ChunkHit],
        cosine_by_id: dict[UUID, float],
        strong: frozenset[UUID],
        *,
        supporting: bool,
    ) -> list[ChunkHit]:
        by_id = {hit.id: hit for hit in hits}
        # Fusion can bury the one passage only the dense arm understands (a question
        # worded unlike the page). Reserve two slots for the nearest by meaning, then
        # fill the rest in fused order. No per-page quota: a single page may hold
        # every relevant fact.
        nearest = [
            chunk_id
            for chunk_id, cosine in sorted(
                cosine_by_id.items(), key=lambda item: (-item[1], str(item[0]))
            )
            if cosine >= COSINE_FLOOR and chunk_id in by_id
        ][:2]
        scores = dict(ranked)
        order = [(chunk_id, scores.get(chunk_id, 0.0)) for chunk_id in nearest] + [
            item for item in ranked if item[0] not in nearest
        ]
        selected: list[ChunkHit] = []
        skipped: list[dict] = []
        limit = EXACT_SUPPORT if supporting else KEEP
        for chunk_id, score in order:
            hit = by_id.get(chunk_id)
            if hit is None:
                continue
            cosine = cosine_by_id.get(chunk_id)
            vouched = chunk_id in strong
            if cosine is not None and cosine < COSINE_FLOOR and not vouched:
                skipped.append({"id": chunk_id, "reason": "below_cosine_floor"})
                continue
            if supporting and not vouched and cosine is None:
                skipped.append({"id": chunk_id, "reason": "weak_support"})
                continue
            if len(selected) >= limit:
                skipped.append({"id": chunk_id, "reason": "document_budget"})
                continue
            selected.append(replace(hit, rrf=score, cosine=cosine))
        record_trace(
            "retrieval",
            fused_ranking=ranked,
            selection_ranking=order,
            selection_skipped=skipped,
            candidates=hits,
        )
        return selected

    async def _exact_match(self, site_id: UUID, visitor_text: str) -> ChunkHit | None:
        key = token_key(visitor_text)
        if len(key) < 3:
            return None
        # Every token of a word-for-word match is in the indexed question or alias, so
        # the text index narrows the candidates; equality is then checked on folded keys.
        tsquery = func.plainto_tsquery(_SIMPLE, fold_text(visitor_text))
        result = await self._session.execute(
            _eligible_join(select(KbChunk, KbPage, KbSource))
            .where(*_eligible(site_id), KbChunk.search_document.op("@@")(tsquery))
            .order_by(func.ts_rank_cd(KbChunk.search_document, tsquery).desc(), KbChunk.id)
            .limit(EXACT_CANDIDATES)
        )
        matches: list[ChunkHit] = []
        for chunk, page, source in result.all():
            if contains_injection_marker(chunk.answer_verbatim) or contains_injection_marker(
                chunk.heading or ""
            ):
                continue
            if is_marketing_cta(chunk.answer_verbatim):
                continue
            candidates = [chunk.canonical_question or "", *(chunk.aliases or [])]
            if key not in {token_key(item) for item in candidates if item}:
                continue
            matches.append(self._to_hit(chunk, page, source, rrf=1.0))
        return matches[0] if len(matches) == 1 else None

    async def _fts(self, site_id: UUID, visitor_text: str) -> _Arm:
        tokens = search_tokens(visitor_text)[:MAX_QUERY_TOKENS]
        if not tokens:
            return _NO_ARM
        any_term = func.to_tsquery(_SIMPLE, tsquery_text(tokens, "|"))
        all_terms = func.to_tsquery(_SIMPLE, tsquery_text(tokens, "&"))
        matches_all = KbChunk.search_document.op("@@")(all_terms)
        # Chunks holding every term lead; the rest are ordered by cover density, so
        # coverage of the question is rewarded without a language-specific stop list.
        rank = func.ts_rank_cd(KbChunk.search_document, any_term, 1)
        result = await self._session.execute(
            _eligible_join(
                select(
                    KbChunk.id,
                    rank,
                    matches_all,
                    KbChunk.heading,
                    KbChunk.answer_verbatim,
                    KbChunk.aliases,
                )
            )
            .where(*_eligible(site_id), KbChunk.search_document.op("@@")(any_term))
            .order_by(matches_all.desc(), rank.desc(), KbChunk.id)
            .limit(FTS_LIMIT)
        )
        scored: list[tuple[UUID, float]] = []
        strong: set[UUID] = set()
        for chunk_id, score, all_match, heading, answer, aliases in result.all():
            blob = f"{heading or ''} {answer or ''} {' '.join(aliases or [])}"
            if contains_injection_marker(blob):
                continue
            scored.append((chunk_id, float(score or 0.0)))
            if all_match and len(tokens) >= MIN_STRONG_TOKENS:
                strong.add(chunk_id)
        return _Arm(scored, frozenset(strong))

    async def _trigram(self, site_id: UUID, visitor_text: str) -> _Arm:
        query = (visitor_text or "").strip()
        if len(query) < 3:
            return _NO_ARM
        await self._session.execute(text("SET LOCAL pg_trgm.similarity_threshold = 0.30"))
        sql = text(
            f"""
            SELECT kb_chunks.id,
                   similarity({_TRGM_EXPR}, workspace_search_fold(:q)) AS score,
                   kb_chunks.heading,
                   kb_chunks.answer_verbatim,
                   kb_chunks.aliases
              FROM kb_chunks
              JOIN kb_pages ON kb_pages.id = kb_chunks.page_id
              JOIN kb_snapshots ON kb_snapshots.id = kb_chunks.snapshot_id
              JOIN kb_sources ON kb_sources.id = kb_pages.source_id
             WHERE kb_chunks.site_id = :site_id
               AND kb_chunks.enabled IS TRUE
               AND kb_chunks.kind <> 'refusal'
               AND kb_pages.enabled IS TRUE
               AND kb_sources.enabled IS TRUE
               AND kb_snapshots.state = 'live'
               AND {_TRGM_EXPR} % workspace_search_fold(:q)
             ORDER BY score DESC, kb_chunks.id
             LIMIT :fts_limit
            """
        )
        result = await self._session.execute(
            sql, {"q": query, "site_id": site_id, "fts_limit": FTS_LIMIT}
        )
        scored: list[tuple[UUID, float]] = []
        strong: set[UUID] = set()
        for chunk_id, score, heading, answer, aliases in result.all():
            blob = f"{heading or ''} {answer or ''} {' '.join(aliases or [])}"
            if contains_injection_marker(blob):
                continue
            scored.append((chunk_id, float(score or 0.0)))
            if float(score or 0.0) >= TRGM_STRONG:
                strong.add(chunk_id)
        return _Arm(scored, frozenset(strong))

    async def _dense(
        self, site_id: UUID, query_vector: list[float]
    ) -> tuple[list[UUID], dict[UUID, float]]:
        await self._tune_ann()
        distance = KbChunk.embedding.cosine_distance(query_vector)
        live = select(KbSnapshot.id).where(
            KbSnapshot.site_id == site_id, KbSnapshot.state == "live"
        )
        # Chunk columns only, ordered by distance alone: this is the shape the HNSW
        # index serves. Page and source switches filter the small window afterwards.
        nearest = (
            select(
                KbChunk.id.label("id"),
                KbChunk.page_id.label("page_id"),
                KbChunk.site_id.label("site_id"),
                KbChunk.heading.label("heading"),
                KbChunk.answer_verbatim.label("answer_verbatim"),
                distance.label("distance"),
            )
            .where(
                KbChunk.site_id == site_id,
                KbChunk.enabled.is_(True),
                KbChunk.kind != "refusal",
                KbChunk.embedding.is_not(None),
                KbChunk.snapshot_id.in_(live),
            )
            .order_by(distance)
            .limit(DENSE_OVERFETCH)
            .subquery("nearest")
        )
        result = await self._session.execute(
            select(
                nearest.c.id,
                nearest.c.site_id,
                nearest.c.heading,
                nearest.c.answer_verbatim,
                (1 - nearest.c.distance).label("cosine"),
            )
            .join(KbPage, KbPage.id == nearest.c.page_id)
            .join(KbSource, KbSource.id == KbPage.source_id)
            .where(KbPage.enabled.is_(True), KbSource.enabled.is_(True))
            .order_by(nearest.c.distance, nearest.c.id)
            .limit(DENSE_LIMIT)
        )
        ids: list[UUID] = []
        scores: dict[UUID, float] = {}
        for chunk_id, row_site, heading, answer, cosine in result.all():
            if row_site != site_id:
                return [], {}
            if contains_injection_marker(f"{heading or ''} {answer or ''}"):
                continue
            ids.append(chunk_id)
            scores[chunk_id] = float(cosine)
        return ids, scores

    async def _tune_ann(self) -> None:
        """Let HNSW keep scanning until the site filter is satisfied (pgvector >= 0.8)."""
        global _ann_iterative_scan
        if _ann_iterative_scan is None:
            version = await self._session.scalar(
                text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            )
            parts = [int(part) for part in (version or "0").split(".")[:2] if part.isdigit()]
            _ann_iterative_scan = tuple(parts) >= (0, 8)
        if _ann_iterative_scan:
            await self._session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
            await self._session.execute(text(f"SET LOCAL hnsw.ef_search = {HNSW_EF_SEARCH}"))

    async def _cosines(
        self, site_id: UUID, chunk_ids: list[UUID], query_vector: list[float]
    ) -> dict[UUID, float]:
        if not chunk_ids:
            return {}
        result = await self._session.execute(
            select(KbChunk.id, 1 - KbChunk.embedding.cosine_distance(query_vector)).where(
                KbChunk.id.in_(chunk_ids),
                KbChunk.site_id == site_id,
                KbChunk.embedding.is_not(None),
            )
        )
        return {chunk_id: float(cosine) for chunk_id, cosine in result.all()}

    async def _neighbour_cosines(
        self, site_id: UUID, anchor_id: UUID, chunk_ids: list[UUID]
    ) -> dict[UUID, float]:
        """Cosine of each chunk to the embedding of the ``anchor_id`` chunk, same site only."""
        if not chunk_ids:
            return {}
        anchor = (
            await self._session.execute(
                select(KbChunk.embedding).where(
                    KbChunk.id == anchor_id,
                    KbChunk.site_id == site_id,
                    KbChunk.embedding.is_not(None),
                )
            )
        ).scalar_one_or_none()
        if anchor is None:
            return {}
        return await self._cosines(site_id, chunk_ids, list(anchor))

    async def _start_pages(self, site_id: UUID, query_vector: list[float] | None) -> list[ChunkHit]:
        """The site's landing content for an overview: start pages first, then one passage
        from each other page (a brief start page only introduces the rest)."""
        starts = await self._session.execute(
            _eligible_join(select(KbChunk.id))
            .where(*_eligible(site_id), KbPage.url == KbSource.start_url)
            .order_by(KbPage.url, KbChunk.ordinal, KbChunk.id)
            .limit(FTS_LIMIT)
        )
        ids = list(starts.scalars().all())
        if len(ids) < KEEP:
            others = await self._session.execute(
                _eligible_join(select(KbChunk.id))
                .where(*_eligible(site_id), KbPage.url != KbSource.start_url)
                .ext(distinct_on(KbChunk.page_id))
                .order_by(KbChunk.page_id, KbChunk.ordinal, KbChunk.id)
                .limit(FTS_LIMIT)
            )
            ids.extend(others.scalars().all())
        hits = {hit.id: hit for hit in await self._load(site_id, ids)}
        ordered = [hits[chunk_id] for chunk_id in ids if chunk_id in hits]
        if query_vector is not None:
            cosines = await self._cosines(site_id, [hit.id for hit in ordered], query_vector)
            ordered = [replace(hit, cosine=cosines.get(hit.id)) for hit in ordered]
        return ordered

    async def _load(self, site_id: UUID, chunk_ids: list[UUID]) -> list[ChunkHit]:
        if not chunk_ids:
            return []
        result = await self._session.execute(
            select(KbChunk, KbPage, KbSource)
            .join(KbPage, KbPage.id == KbChunk.page_id)
            .join(KbSnapshot, KbSnapshot.id == KbChunk.snapshot_id)
            .join(KbSource, KbSource.id == KbPage.source_id)
            .where(
                KbChunk.id.in_(chunk_ids),
                KbChunk.site_id == site_id,
                KbChunk.enabled.is_(True),
                KbPage.enabled.is_(True),
                KbSource.enabled.is_(True),
                KbSnapshot.state == "live",
            )
        )
        hits: list[ChunkHit] = []
        for chunk, page, source in result.all():
            if chunk.site_id != site_id:
                return []
            if contains_injection_marker(chunk.answer_verbatim) or contains_injection_marker(
                chunk.heading or ""
            ):
                continue
            if is_marketing_cta(chunk.answer_verbatim):
                continue
            hits.append(self._to_hit(chunk, page, source, rrf=0.0))
        return hits

    @staticmethod
    def _to_hit(chunk: KbChunk, page: KbPage, source: KbSource, *, rrf: float) -> ChunkHit:
        origins = [url for url in (chunk.origin_urls or []) if url]
        page_url = page.public_url
        start_url = source.start_url
        url = visitor_citation_url(page_url=page_url, origin_urls=origins, start_url=start_url)
        title = urlparse(start_url).netloc if len(origins) >= 2 else page.title
        return ChunkHit(
            id=chunk.id,
            page_id=chunk.page_id,
            site_id=chunk.site_id,
            heading=chunk.heading,
            body=chunk.body,
            url=url or page_url,
            title=title or page.title,
            rrf=rrf,
            cosine=None,
            answer_verbatim=chunk.answer_verbatim,
            display_locator=None,
            snapshot_id=chunk.snapshot_id,
            canonical_question=chunk.canonical_question,
            legal_sensitive=bool(chunk.legal_sensitive),
            aliases=tuple(chunk.aliases or []),
            topic_label=chunk.topic_label or chunk.heading,
            risk_class=chunk.risk_class or "general",
            answer_mode=chunk.answer_mode or "paraphrase_allowed",
            enabled=bool(chunk.enabled),
            structured=bool(chunk.context_prefix),
        )
