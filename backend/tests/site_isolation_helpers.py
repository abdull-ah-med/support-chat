"""Two-site fixtures for the isolation tests. Test doubles live here, never in app/.

Every helper is brand-neutral: sites are "alpha" and "beta", and they deliberately share the
same question text, shortcuts and visitor phrasing so any missing site filter shows up as a
wrong-site row instead of hiding behind distinct wording.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation
from app.models.kb_chunk import KbChunk
from app.models.kb_page import KbPage
from app.models.kb_snapshot import KbSnapshot
from app.models.kb_source import KbSource
from app.models.site import Site
from app.models.visitor import Visitor
from app.services.grounded_response_types import Citation, ModelDraft
from app.settings import get_settings
from tests.ws_helpers import HOST_ORIGIN

ALPHA_KEY = "alpha"
BETA_KEY = "beta"
SHARED_QUESTION = "How do I reset my account password?"
SHARED_ANSWER = "Open Settings, choose Security, then select Reset password."
ALPHA_ONLY_QUESTION = "Which payment methods does Alpha accept?"
ALPHA_ONLY_ANSWER = "Alpha accepts cards and bank transfer."
BETA_ONLY_QUESTION = "Does Beta offer on-site installation?"
BETA_ONLY_ANSWER = "Beta offers on-site installation within two weeks."

# Topic -> axis. Texts that mention a topic share a unit vector, everything else is orthogonal
# to every topic. That makes cosine similarity exactly 1.0 for "same topic" and 0.0 otherwise.
TOPIC_AXES = {
    "password": 0,
    "payment": 1,
    "installation": 2,
    "invoice": 3,
    "hours": 4,
}
OTHER_AXIS = 50


def _axis(text: str) -> int:
    lowered = text.casefold()
    for topic, axis in TOPIC_AXES.items():
        if topic in lowered:
            return axis
    return OTHER_AXIS


class TopicEmbedder:
    """Deterministic test double for the Embedder protocol."""

    embedder_id = "test:topics"

    @property
    def dim(self) -> int:
        return get_settings().openai_embed_dim

    def vector(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        vector[_axis(text)] = 1.0
        return vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float] | None:
        return self.vector(text)


@dataclass
class ForeignCitingResponder:
    """Grounded-draft double. Cites the first offered unit, or a chosen id (a hostile model)."""

    cite_chunk_id: uuid.UUID | None = None
    cite_snapshot_id: uuid.UUID | None = None
    # True: answer only from an offered unit whose question is the visitor's question, and
    # otherwise return an uncited reply (a model with nothing to cite).
    match_question: bool = False
    calls: list[dict] = field(default_factory=list)

    @property
    def offered_ids(self) -> set[uuid.UUID]:
        return {unit_id for call in self.calls for unit_id in call["unit_ids"]}

    @property
    def offered_text(self) -> str:
        return "\n".join(text for call in self.calls for text in call["texts"])

    def _pick(self, visitor_text: str, documents):
        if not self.match_question:
            return documents[0]
        wanted = visitor_text.casefold().strip()
        for unit in documents:
            if (unit.canonical_question or "").casefold().strip() == wanted:
                return unit
        return None

    async def understand_turn(self, inputs):
        """Reads every message as a plain knowledge question, as a model would for these."""
        from app.llm.understanding import TurnUnderstanding

        return TurnUnderstanding(
            language="en",
            intent="question",
            relation="standalone",
            query=inputs.latest_message,
            ambiguity="",
            matches_handoff_topic=False,
            reply="",
        )

    async def generate_grounded_draft(self, turn, documents, **_kwargs):
        self.calls.append(
            {
                "visitor_text": turn.visitor_text,
                "unit_ids": [unit.id for unit in documents],
                "texts": [unit.answer_verbatim for unit in documents],
            }
        )
        if self.cite_chunk_id is not None:
            quote = "foreign evidence"
            chunk_id, snapshot_id, title, url = (
                self.cite_chunk_id,
                self.cite_snapshot_id,
                "Foreign",
                "https://foreign.test/page",
            )
        elif documents and (unit := self._pick(turn.visitor_text, documents)) is not None:
            quote = unit.answer_verbatim
            chunk_id, snapshot_id = unit.id, unit.snapshot_id
            title, url = unit.source_title, unit.source_url
        else:
            return ModelDraft(body="I am not sure about that.", citations=[], request_id="req_none")
        body = f"Yes. {quote}"
        return ModelDraft(
            body=body,
            citations=[
                Citation(
                    chunk_id=chunk_id,
                    snapshot_id=snapshot_id,
                    response_start=0,
                    response_end=len(body),
                    source_start=0,
                    source_end=len(quote),
                    cited_text=quote,
                    source_title=title,
                    source_url=url,
                )
            ],
            request_id="req_isolation",
        )


async def add_site(session: AsyncSession, key: str, name: str) -> Site:
    site = Site(
        key=key,
        name=name,
        public_key=uuid.uuid4().hex + uuid.uuid4().hex,
        allowed_origins=[HOST_ORIGIN],
        greeting="Talk to a specialist.",
        privacy_url="http://localhost:3000/privacy",
    )
    session.add(site)
    await session.flush()
    return site


async def add_live_chunk(
    session: AsyncSession,
    site: Site,
    question: str,
    answer: str,
    *,
    url: str | None = None,
    embedder: TopicEmbedder | None = None,
) -> KbChunk:
    """One live chunk (own source, page and snapshot) so sites never share a parent row."""
    embedder = embedder or TopicEmbedder()
    page_url = url or f"https://{site.key}.test/{uuid.uuid4().hex}"
    source = KbSource(
        site_id=site.id,
        start_url=page_url,
        mode="list",
        seed_urls=[page_url],
        status="ready",
        page_count=1,
        embedder_id=embedder.embedder_id,
        source_kind="legacy_faq",
        enabled=True,
    )
    session.add(source)
    await session.flush()
    digest = hashlib.sha256(answer.encode()).hexdigest()
    page = KbPage(
        source_id=source.id,
        site_id=site.id,
        url=page_url,
        title=question,
        content_text=answer,
        content_sha256=digest,
        http_status=200,
        enabled=True,
    )
    session.add(page)
    await session.flush()
    snapshot = KbSnapshot(
        site_id=site.id,
        source_id=source.id,
        state="live",
        content_hash=digest,
        token_estimate=0,
    )
    session.add(snapshot)
    await session.flush()
    chunk = KbChunk(
        page_id=page.id,
        site_id=site.id,
        snapshot_id=snapshot.id,
        ordinal=0,
        kind="prose",
        heading=question,
        canonical_question=question,
        answer_verbatim=answer,
        aliases=[],
        body=answer,
        embedding=embedder.vector(f"{question}\n{answer}"),
        enabled=True,
    )
    session.add(chunk)
    await session.flush()
    return chunk


async def add_bot_conversation(session: AsyncSession, site: Site) -> tuple[Visitor, Conversation]:
    visitor = Visitor(
        site_id=site.id,
        resume_token_hash=uuid.uuid4().hex,
        name="Sam Visitor",
        email=f"sam@{site.key}.test",
    )
    session.add(visitor)
    await session.flush()
    conversation = Conversation(site_id=site.id, visitor_id=visitor.id, state="bot")
    session.add(conversation)
    await session.flush()
    return visitor, conversation


@dataclass
class TwoSites:
    alpha: Site
    beta: Site
    alpha_shared: KbChunk
    beta_shared: KbChunk
    alpha_only: KbChunk
    beta_only: KbChunk


async def seed_two_sites(session: AsyncSession) -> TwoSites:
    """Both sites publish the same password FAQ verbatim, plus one private FAQ each."""
    alpha = await add_site(session, ALPHA_KEY, "Alpha Co")
    beta = await add_site(session, BETA_KEY, "Beta Co")
    seeded = TwoSites(
        alpha=alpha,
        beta=beta,
        alpha_shared=await add_live_chunk(session, alpha, SHARED_QUESTION, SHARED_ANSWER),
        beta_shared=await add_live_chunk(session, beta, SHARED_QUESTION, SHARED_ANSWER),
        alpha_only=await add_live_chunk(session, alpha, ALPHA_ONLY_QUESTION, ALPHA_ONLY_ANSWER),
        beta_only=await add_live_chunk(session, beta, BETA_ONLY_QUESTION, BETA_ONLY_ANSWER),
    )
    await session.commit()
    return seeded
