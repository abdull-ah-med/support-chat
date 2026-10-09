"""Build small knowledge corpora with exactly the fields a retrieval test cares about."""

import hashlib
from dataclasses import dataclass, field
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.kb_chunk import KbChunk
from app.models.kb_page import KbPage
from app.models.kb_snapshot import KbSnapshot
from app.models.kb_source import KbSource
from app.models.site import Site
from app.services.kb_embedder import configured_embedder_id


@dataclass
class ChunkSpec:
    answer: str
    heading: str = ""
    question: str | None = None
    aliases: list[str] = field(default_factory=list)
    body: str | None = None
    vector: list[float] | None = None
    kind: str = "prose"


@dataclass
class Corpus:
    source: KbSource
    snapshot: KbSnapshot
    pages: list[KbPage]
    chunks: list[list[KbChunk]]


async def make_corpus(
    session: AsyncSession,
    site: Site,
    pages: list[list[ChunkSpec]],
    *,
    start_url: str | None = None,
    state: str = "live",
) -> Corpus:
    """One source and one snapshot holding ``pages``; page ``i`` has ``pages[i]`` chunks."""
    base = start_url or f"https://{site.key}.test/"
    source = KbSource(
        site_id=site.id,
        start_url=base,
        mode="list",
        seed_urls=[base],
        status="ready",
        page_count=len(pages),
        embedder_id=configured_embedder_id(),
        source_kind="legacy_faq",
        enabled=True,
    )
    session.add(source)
    await session.flush()
    snapshot = KbSnapshot(
        site_id=site.id,
        source_id=source.id,
        state=state,
        content_hash=hashlib.sha256(base.encode()).hexdigest(),
        token_estimate=0,
    )
    session.add(snapshot)
    await session.flush()
    built_pages: list[KbPage] = []
    built_chunks: list[list[KbChunk]] = []
    for index, specs in enumerate(pages):
        url = base if index == 0 else f"{base}page-{index}"
        page = KbPage(
            source_id=source.id,
            site_id=site.id,
            url=url,
            title=f"Page {index}",
            content_text=" ".join(spec.answer for spec in specs),
            content_sha256=hashlib.sha256(url.encode()).hexdigest(),
            http_status=200,
            enabled=True,
        )
        session.add(page)
        await session.flush()
        rows: list[KbChunk] = []
        for ordinal, spec in enumerate(specs):
            chunk = KbChunk(
                page_id=page.id,
                site_id=site.id,
                snapshot_id=snapshot.id,
                ordinal=ordinal,
                kind=spec.kind,
                heading=spec.heading,
                canonical_question=spec.question,
                answer_verbatim=spec.answer,
                aliases=list(spec.aliases),
                body=spec.body if spec.body is not None else spec.answer,
                embedding=spec.vector,
                enabled=True,
            )
            session.add(chunk)
            rows.append(chunk)
        await session.flush()
        built_pages.append(page)
        built_chunks.append(rows)
    return Corpus(source, snapshot, built_pages, built_chunks)


@dataclass(frozen=True)
class LiveUnit:
    id: UUID
    answer_verbatim: str
    url: str
    heading: str


async def load_live_units(session: AsyncSession, snapshot_ids: list[UUID]) -> list[LiveUnit]:
    """What a live snapshot currently exposes: enabled chunks of enabled pages and sources."""
    if not snapshot_ids:
        return []
    result = await session.execute(
        select(KbChunk, KbPage)
        .join(KbPage, KbPage.id == KbChunk.page_id)
        .join(KbSnapshot, KbSnapshot.id == KbChunk.snapshot_id)
        .join(KbSource, KbSource.id == KbPage.source_id)
        .where(
            KbChunk.snapshot_id.in_(snapshot_ids),
            KbChunk.enabled.is_(True),
            KbChunk.kind != "refusal",
            KbPage.enabled.is_(True),
            KbSnapshot.state == "live",
            KbSource.enabled.is_(True),
        )
        .order_by(KbChunk.page_id, KbChunk.ordinal, KbChunk.id)
    )
    return [
        LiveUnit(chunk.id, chunk.answer_verbatim, page.public_url, chunk.heading)
        for chunk, page in result.all()
    ]
