"""The engine drafts from evidence, validates structure, repairs once, else says "unknown".

Oracles are the stated rules: a citation must quote this site's live passage exactly; a numeric
claim must come from the cited text or the visitor's words; the drafting model's sentinel and
any failed repair become the honest unknown reply, never invented text.
"""

import uuid
from dataclasses import replace

from app.chat.outcome_copy import INSUFFICIENT_HUMAN, INSUFFICIENT_SOLO, TECH_FAIL_HUMAN
from app.llm.prompts import NO_ANSWER
from app.llm.understanding import TurnUnderstanding
from app.services.grounded_response import (
    Citation,
    EvidenceUnit,
    GroundedResponseEngine,
    ModelDraft,
    ProviderStatus,
    ResponseOutcome,
    TurnContext,
    unknown_decision,
)
from tests.fake_anthropic import understanding

SERVICES_EVIDENCE = EvidenceUnit(
    id=uuid.UUID("10000000-0000-4000-8000-000000000001"),
    canonical_question="Which services are offered?",
    aliases=(),
    topic_label="Services",
    answer_verbatim="We support onsite testing, random pool management, and physicals.",
    source_title="Services",
    source_url="https://example.test/services",
    snapshot_id=uuid.UUID("20000000-0000-4000-8000-000000000001"),
)

TIMING_EVIDENCE = EvidenceUnit(
    id=uuid.UUID("10000000-0000-4000-8000-000000000002"),
    canonical_question="How quickly are results available?",
    aliases=(),
    topic_label="Turnaround",
    answer_verbatim="Results are reported in 24 to 48 hours.",
    source_title="Turnaround",
    source_url="https://example.test/timing",
    snapshot_id=uuid.UUID("20000000-0000-4000-8000-000000000002"),
)


def _citation(unit: EvidenceUnit, body: str) -> Citation:
    return Citation(
        chunk_id=unit.id,
        snapshot_id=unit.snapshot_id,
        response_start=0,
        response_end=len(body),
        source_start=0,
        source_end=len(unit.answer_verbatim),
        cited_text=unit.answer_verbatim,
        source_title=unit.source_title,
        source_url=unit.source_url,
    )


def _turn(*units: EvidenceUnit, text: str = "How long?", **fields) -> TurnContext:
    return TurnContext(
        visitor_text=text,
        evidence=list(units),
        understanding=TurnUnderstanding.model_validate(understanding(query=text)),
        **fields,
    )


def _drafting(draft: ModelDraft | None):
    async def complete(_turn, _units):
        return draft

    return complete


async def test_paraphrased_cited_answer_is_accepted() -> None:
    body = "Results are reported within 24 to 48 hours."
    engine = GroundedResponseEngine(_drafting(ModelDraft(body, [_citation(TIMING_EVIDENCE, body)])))
    decision = await engine.respond(_turn(TIMING_EVIDENCE))

    assert decision.outcome is ResponseOutcome.SYNTHESIZED_ANSWER
    assert decision.body == body
    assert decision.citations[0].chunk_id == TIMING_EVIDENCE.id
    assert decision.provider_status is ProviderStatus.OK
    assert decision.reason_code is None


async def test_forged_source_text_with_a_valid_chunk_id_is_rejected() -> None:
    body = "Results are guaranteed in 12 hours."
    forged = replace(_citation(TIMING_EVIDENCE, body), cited_text=body)
    decision = await GroundedResponseEngine(_drafting(ModelDraft(body, [forged]))).respond(
        _turn(TIMING_EVIDENCE)
    )
    assert decision.reason_code == "grounding_reject"
    assert decision.body.startswith(INSUFFICIENT_HUMAN)
    assert "guaranteed" not in decision.body


async def test_citation_to_a_passage_that_was_not_provided_is_rejected() -> None:
    body = "We support onsite testing."
    decision = await GroundedResponseEngine(
        _drafting(ModelDraft(body, [_citation(SERVICES_EVIDENCE, body)]))
    ).respond(_turn(TIMING_EVIDENCE))
    assert decision.reason_code == "grounding_reject"


async def test_citation_from_a_stale_snapshot_is_rejected() -> None:
    body = "Results are reported in 24 to 48 hours."
    stale = replace(_citation(TIMING_EVIDENCE, body), snapshot_id=uuid.uuid4())
    decision = await GroundedResponseEngine(_drafting(ModelDraft(body, [stale]))).respond(
        _turn(TIMING_EVIDENCE)
    )
    assert decision.reason_code == "grounding_reject"


async def test_answer_without_any_citation_is_rejected() -> None:
    decision = await GroundedResponseEngine(
        _drafting(ModelDraft("Results come quickly.", []))
    ).respond(_turn(TIMING_EVIDENCE))
    assert decision.reason_code == "grounding_reject"
    assert decision.citations == []


async def test_unsupported_number_is_rejected_but_a_number_from_the_visitor_is_not() -> None:
    body = "Results are reported within 72 hours."
    draft = ModelDraft(body, [_citation(TIMING_EVIDENCE, body)])
    rejected = await GroundedResponseEngine(_drafting(draft)).respond(_turn(TIMING_EVIDENCE))
    accepted = await GroundedResponseEngine(_drafting(draft)).respond(
        _turn(TIMING_EVIDENCE, text="Is 72 hours right?")
    )
    assert rejected.reason_code == "grounding_reject"
    assert accepted.outcome is ResponseOutcome.SYNTHESIZED_ANSWER


async def test_digits_in_the_cited_page_title_are_not_invented_numbers() -> None:
    unit = replace(
        SERVICES_EVIDENCE,
        source_title="Random testing | 247 SampleSite",
        answer_verbatim="We manage the pool and generate the unannounced selections.",
    )
    body = "247 SampleSite manages the pool and generates the unannounced selections."
    draft = ModelDraft(body, [_citation(unit, body)])
    decision = await GroundedResponseEngine(_drafting(draft)).respond(_turn(unit))
    assert decision.outcome is ResponseOutcome.SYNTHESIZED_ANSWER

    uncited_title = replace(unit, source_title="Random testing")
    rejected = await GroundedResponseEngine(_drafting(draft)).respond(_turn(uncited_title))
    assert rejected.reason_code == "grounding_reject"


async def test_numbers_match_across_scripts_and_thousands_separators() -> None:
    unit = replace(TIMING_EVIDENCE, answer_verbatim="El costo es 1.000 al mes y 24 horas.")
    body = "El costo es 1,000 al mes y ٢٤ horas."
    draft = ModelDraft(body, [_citation(unit, body)])
    decision = await GroundedResponseEngine(_drafting(draft)).respond(_turn(unit))
    assert decision.outcome is ResponseOutcome.SYNTHESIZED_ANSWER


async def test_pasted_passages_are_not_an_answer() -> None:
    body = f"{SERVICES_EVIDENCE.answer_verbatim}\n\n{TIMING_EVIDENCE.answer_verbatim}"
    decision = await GroundedResponseEngine(
        _drafting(ModelDraft(body, [_citation(SERVICES_EVIDENCE, body)]))
    ).respond(_turn(SERVICES_EVIDENCE, TIMING_EVIDENCE))
    assert decision.reason_code == "grounding_reject"


async def test_contact_literal_the_model_did_not_cite_is_attributed_to_evidence() -> None:
    unit = replace(SERVICES_EVIDENCE, answer_verbatim="Write to help@example.test for quotes.")
    body = "Write to help@example.test for quotes."
    cited = replace(_citation(unit, body), response_end=5, source_end=5, cited_text="Write")
    decision = await GroundedResponseEngine(_drafting(ModelDraft(body, [cited]))).respond(
        _turn(unit)
    )
    assert decision.outcome is ResponseOutcome.SYNTHESIZED_ANSWER
    assert any("help@example.test" in c.cited_text for c in decision.citations)


async def test_contact_literal_not_in_evidence_is_never_shown() -> None:
    body = "Write to nobody@example.test."
    decision = await GroundedResponseEngine(
        _drafting(ModelDraft(body, [_citation(TIMING_EVIDENCE, body)]))
    ).respond(_turn(TIMING_EVIDENCE))
    assert decision.reason_code == "grounding_reject"
    assert "nobody@example.test" not in decision.body


# ---- unknown paths -------------------------------------------------------------------------


async def test_no_evidence_never_calls_the_drafting_model() -> None:
    calls = []

    async def complete(_turn, _units):
        calls.append(1)

    decision = await GroundedResponseEngine(complete).respond(_turn())
    assert calls == []
    assert decision.reason_code == "no_evidence"
    assert decision.outcome is ResponseOutcome.KNOWLEDGE_GAP
    assert decision.body == INSUFFICIENT_HUMAN
    assert decision.offer_handoff is True


async def test_disabled_or_human_only_evidence_counts_as_no_evidence() -> None:
    hidden = replace(TIMING_EVIDENCE, answer_mode="human_only")
    decision = await GroundedResponseEngine(_drafting(ModelDraft("x", []))).respond(_turn(hidden))
    assert decision.reason_code == "no_evidence"


async def test_sentinel_is_the_unknown_reply_not_text() -> None:
    decision = await GroundedResponseEngine(_drafting(ModelDraft(NO_ANSWER, []))).respond(
        _turn(TIMING_EVIDENCE)
    )
    assert decision.reason_code == "no_evidence"
    assert NO_ANSWER not in decision.body
    assert decision.body.startswith(INSUFFICIENT_HUMAN)


async def test_second_miss_in_a_row_is_marked_repeated() -> None:
    decision = await GroundedResponseEngine(_drafting(ModelDraft(NO_ANSWER, []))).respond(
        _turn(TIMING_EVIDENCE, prior_miss_count=1)
    )
    assert decision.reason_code == "repeated_miss"
    assert decision.offer_handoff is True


async def test_unknown_reply_without_staff_shows_the_contact_route_and_offers_no_person() -> None:
    decision = await GroundedResponseEngine(_drafting(ModelDraft(NO_ANSWER, []))).respond(
        _turn(TIMING_EVIDENCE, human_enabled=False, contact_info=("help@example.test",))
    )
    assert decision.body == f"{INSUFFICIENT_SOLO}\n\nhelp@example.test"
    assert decision.offer_handoff is False


async def test_model_written_unknown_text_replaces_the_neutral_line() -> None:
    turn = _turn(TIMING_EVIDENCE, human_enabled=False, contact_info=("help@example.test",))
    decision = unknown_decision(turn, text="No tengo informacion verificada sobre eso.")
    assert decision.body == "No tengo informacion verificada sobre eso.\n\nhelp@example.test"


async def test_unknown_text_may_repeat_digits_of_the_site_name() -> None:
    turn = _turn(TIMING_EVIDENCE, site_name="247 Example")
    decision = unknown_decision(
        turn, text="I have no verified information from 247 Example on that."
    )
    assert decision.body.startswith("I have no verified information from 247 Example")


async def test_unusable_model_unknown_text_falls_back_to_neutral_copy() -> None:
    staffed = unknown_decision(
        TurnContext(visitor_text="q", evidence=[]), text="See https://example.test for details"
    )
    solo = unknown_decision(
        TurnContext(visitor_text="q", evidence=[], human_enabled=False),
        text="Call 555 0199 now",
    )
    assert staffed.body == INSUFFICIENT_HUMAN
    assert solo.body == INSUFFICIENT_SOLO
    assert "specialist" not in INSUFFICIENT_SOLO


async def test_pending_handoff_means_no_new_offer_of_a_person() -> None:
    decision = unknown_decision(TurnContext(visitor_text="q", evidence=[], handoff_pending=True))
    assert decision.offer_handoff is False


# ---- repair ----------------------------------------------------------------------------------


async def test_failed_validation_gets_exactly_one_repair() -> None:
    bad = ModelDraft("Results come quickly.", [])
    good_body = "Results are reported in 24 to 48 hours."
    good = ModelDraft(good_body, [_citation(TIMING_EVIDENCE, good_body)])
    seen = []

    async def repair(_turn, _units, draft, failure):
        seen.append((draft.body, failure))
        return good

    decision = await GroundedResponseEngine(_drafting(bad), repair).respond(_turn(TIMING_EVIDENCE))
    assert decision.outcome is ResponseOutcome.SYNTHESIZED_ANSWER
    assert decision.repair_used is True
    assert seen == [("Results come quickly.", "no_citation")]


async def test_failed_repair_ends_in_the_unknown_reply_and_does_not_retry() -> None:
    bad = ModelDraft("Results come quickly.", [])
    attempts = []

    async def repair(_turn, _units, _draft, _failure):
        attempts.append(1)
        return bad

    decision = await GroundedResponseEngine(_drafting(bad), repair).respond(_turn(TIMING_EVIDENCE))
    assert attempts == [1]
    assert decision.reason_code == "grounding_reject"
    assert decision.repair_used is True


async def test_repair_that_finds_nothing_supported_is_the_unknown_reply() -> None:
    async def repair(_turn, _units, _draft, _failure):
        return ModelDraft(NO_ANSWER, [])

    decision = await GroundedResponseEngine(
        _drafting(ModelDraft("Results come quickly.", [])), repair
    ).respond(_turn(TIMING_EVIDENCE))
    assert decision.reason_code == "grounding_reject"


# ---- provider failures ------------------------------------------------------------------------


async def test_provider_exception_is_an_honest_technical_failure() -> None:
    async def complete(_turn, _units):
        raise TimeoutError

    decision = await GroundedResponseEngine(complete).respond(_turn(TIMING_EVIDENCE))
    assert decision.body == TECH_FAIL_HUMAN
    assert decision.provider_status is ProviderStatus.TECH_FAIL
    assert decision.reason_code == "tech_fail"


async def test_empty_or_missing_draft_is_a_technical_failure_never_invented_text() -> None:
    for draft in (None, ModelDraft("   ", [])):
        decision = await GroundedResponseEngine(_drafting(draft)).respond(_turn(TIMING_EVIDENCE))
        assert decision.provider_status is ProviderStatus.TECH_FAIL
        assert decision.citations == []


async def test_missing_provider_is_a_technical_failure() -> None:
    decision = await GroundedResponseEngine(None).respond(_turn(TIMING_EVIDENCE))
    assert decision.provider_status is ProviderStatus.TECH_FAIL
