"""Oracles from the September 18 live report and original public page wording."""

from dataclasses import replace
from uuid import uuid4

from app.db import session_maker
from app.services.grounded_response import (
    Citation,
    EvidenceUnit,
    GroundedResponseEngine,
    ModelDraft,
    TurnContext,
)
from app.services.kb_page_structure import PageBlocks, evidence_from_blocks
from tests.bot_fixtures import insert_chunk, insert_site
from tests.fakes import unit_vector


def evidence(text: str) -> EvidenceUnit:
    return EvidenceUnit(
        id=uuid4(),
        canonical_question=None,
        aliases=(),
        topic_label="SampleMail",
        answer_verbatim=text,
        source_title="SampleMail",
        source_url="https://sample-data.example.com/samplemail",
        snapshot_id=uuid4(),
    )


def citation(unit: EvidenceUnit, body: str, claim: str) -> Citation:
    start = body.index(claim)
    return Citation(
        unit.id,
        unit.snapshot_id,
        start,
        start + len(claim),
        0,
        len(unit.answer_verbatim),
        unit.answer_verbatim,
        unit.source_title,
        unit.source_url,
    )


def test_unsupported_non_numeric_ingestion_claim_cannot_be_cited() -> None:
    source = "247 Instant Check is a consumer public-records search service."
    units = evidence_from_blocks(
        PageBlocks.model_validate(
            {
                "blocks": [
                    {
                        "heading": "247 Instant Check",
                        "text": "247 Instant Check provides identity and compliance verification.",
                        "tags": [],
                    }
                ]
            }
        ),
        source,
        "https://sample-data.example.com/",
    )
    assert units[0].enabled is False
    assert units[0].review_note == "unsupported_source_text"


async def test_public_cited_contacts_are_usable_but_uncited_visitor_contacts_are_blocked() -> None:
    unit = evidence("Contact SampleMail at sales@samplemail.example.com or 239-291-9323.")
    body = unit.answer_verbatim

    async def complete(_turn, _units):
        return ModelDraft(body, [citation(unit, body, body)])

    result = await GroundedResponseEngine(complete).respond(
        TurnContext("What is your email?", [unit])
    )
    assert result.body == body
    assert result.citations[0].cited_text == body

    async def leaking(_turn, _units):
        extra = body + " Send it to visitor@example.net."
        return ModelDraft(extra, [replace(citation(unit, extra, body), response_end=len(extra))])

    blocked = await GroundedResponseEngine(leaking).respond(
        TurnContext("How can I contact you?", [unit])
    )
    assert blocked.reason_code == "grounding_reject"
    assert "visitor@example.net" not in blocked.body


async def test_dense_only_coverage_survives_generic_lexical_matches(migrated_db) -> None:
    from app.services.kb_hybrid import HybridKbSearch

    async with session_maker()() as session:
        site = await insert_site(session, "dense-recall", "Sample Data")
        target = await insert_chunk(
            session,
            site,
            "Geographic Coverage",
            "Solutions are available nationwide across all 50 states.",
            slug="coverage",
        )
        target.embedding = unit_vector(0)
        for index in range(10):
            distractor = await insert_chunk(
                session,
                site,
                "Collection solutions",
                "Collection solutions support creditor rights firms with data tools.",
                slug=f"generic-{index}",
            )
            distractor.embedding = unit_vector(1)
        await session.commit()
        hits = await HybridKbSearch(session).search(
            site.id,
            "Do your collection solutions work in every state?",
            query_vector=unit_vector(0),
        )
        assert any(hit.id == target.id for hit in hits)
        assert len(hits) <= 8


def test_default_window_retains_initial_requirements_at_twelfth_turn() -> None:
    from app.models.message import Message
    from app.services.full_context import prior_provider_messages

    rows = [Message(role="visitor", body="We need to locate debtors and mail demand letters.")]
    for _ in range(11):
        rows.extend(
            [
                Message(role="bot", body="SampleMail verifies addresses."),
                Message(role="visitor", body="Tell me more."),
            ]
        )
    history = prior_provider_messages(rows, "Tell me more.")
    assert history[0] == {
        "role": "user",
        "content": "We need to locate debtors and mail demand letters.",
    }


def test_deduplication_keeps_a_distinct_prohibition() -> None:
    from app.services.kb_extract.types import EvidenceUnit as IngestUnit
    from app.services.kb_page_structure import unique_units

    allow = IngestUnit(
        "section", "Permitted uses", None, "Data may be used for employment decisions.", "", None
    )
    forbid = IngestUnit(
        "section", "Restrictions", None, "Data may not be used for employment decisions.", "", None
    )
    assert [u.answer_verbatim for u in unique_units([allow, forbid])] == [
        "Data may be used for employment decisions.",
        "Data may not be used for employment decisions.",
    ]


async def test_website_sections_preserve_contacts_and_prohibitions_without_a_model(
    monkeypatch,
) -> None:
    from app.services.kb_page_structure import HaikuPageStructurer

    def unexpected_provider(**_kwargs):
        raise AssertionError("Website source passages must not depend on paid generative rewriting")

    monkeypatch.setattr("anthropic.AsyncAnthropic", unexpected_provider)
    source = "## Contact\n\nsales@samplemail.example.com or 239-291-9323.\n\n## Restrictions\n\nData is not a consumer report and may not be used for employment decisions."
    result = await HaikuPageStructurer().structure_page(
        "https://sample-data.example.com/", "Home", source
    )
    assert [u.answer_verbatim for u in result.units] == [
        "sales@samplemail.example.com or 239-291-9323.",
        "Data is not a consumer report and may not be used for employment decisions.",
    ]
    assert result.input_tokens == 0


def test_chunk_overlap_preserves_contiguous_source_without_duplicating_tail() -> None:
    from app.services.kb_chunk import pack_chunks
    from app.services.kb_extract.types import EvidenceUnit

    source = (
        ("Addresses are verified before delivery. " * 16).rstrip()
        + "\n\n"
        + ("Coverage spans all 50 states; no delivery date is guaranteed. " * 20)
    )
    unit = EvidenceUnit("section", "Coverage", None, source, source, None)
    chunks = pack_chunks(unit, target=900, overlap=150)
    assert len(chunks) > 1
    assert all(chunk.answer_verbatim.strip() in source for chunk in chunks)


def test_long_crawler_heading_cannot_overflow_provider_title() -> None:
    from app.llm.bot_responder import _document_block

    unit = replace(evidence("Coverage spans all 50 states."), topic_label="Coverage " * 80)
    block = _document_block(unit)
    assert len(block["title"]) <= 200
    assert unit.answer_verbatim in block["source"]["data"]


async def test_native_citation_fragments_keep_supported_contact_sentence() -> None:
    unit = evidence("Phone: 239-955-6596. Support inquiries: support@sample-data.example.com")
    body = "You can also contact support@sample-data.example.com for support inquiries."
    # Native citations can cover a clause rather than its surrounding sentence.
    item = citation(unit, body, "contact support@sample-data.example.com")

    async def complete(_turn, _units):
        return ModelDraft(body, [item])

    result = await GroundedResponseEngine(complete).respond(
        TurnContext("Your support email?", [unit])
    )
    assert result.body == body
    assert result.citations[0].response_start == item.response_start
