"""The eval harness is per site and scores more than substrings.

Scoring tests are pure. The run tests drive a real bot turn on two sites that publish the same
question, with injected doubles for the embedder and model, and use the per-site datasets
`tests/evals/dataset/alpha.json` and `beta.json`.
"""

import json
from uuid import uuid4

import pytest

from app.db import session_maker
from app.services.bot_eval import (
    EvalCase,
    ObservedReply,
    dataset_path_for,
    format_summary,
    load_dataset,
    run_case,
    score_case,
    summarize,
)
from scripts.live_bot_probe import check_chat, probes_path_for
from scripts.run_site_eval import parse_args, require_providers
from tests.site_isolation_helpers import (
    ALPHA_KEY,
    BETA_KEY,
    BETA_ONLY_QUESTION,
    ForeignCitingResponder,
    TopicEmbedder,
    seed_two_sites,
)
from tests.ws_helpers import HOST_ORIGIN

CASE = EvalCase(id="c", family="f.one", turns=("hi",), expected_state="bot")
ANSWER = ObservedReply(
    state="bot",
    role="bot",
    body="Yes.",
    outcome="synthesized_answer",
    citation_chunk_ids=(uuid4(),),
    turn_latencies_ms=(120,),
)
DECLINE = ObservedReply(state="bot", role="system", body="No.", outcome="knowledge_gap")


def _case(**fields) -> EvalCase:
    return EvalCase(id="c", family="f.one", turns=("hi",), expected_state="bot", **fields)


# --- datasets ------------------------------------------------------------------------------


def test_datasets_are_selected_by_site_key(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("app.services.bot_eval.DATASET_DIR", tmp_path)
    for key in ("one", "two"):
        (tmp_path / f"{key}.json").write_text(
            json.dumps({"site_key": key, "cases": [{**_raw("a"), "id": f"{key}.a"}]})
        )
    assert dataset_path_for("one") == tmp_path / "one.json"
    assert [case.id for case in load_dataset("one")[1]] == ["one.a"]
    assert [case.id for case in load_dataset("two")[1]] == ["two.a"]


def test_explicit_dataset_path_overrides_the_default(tmp_path) -> None:
    path = tmp_path / "custom.json"
    path.write_text(json.dumps({"cases": [_raw("x")]}))
    assert load_dataset("any-site", path)[0] == "any-site"
    assert load_dataset(None, path)[0] == "custom"


def test_a_dataset_for_another_site_is_refused(tmp_path) -> None:
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"site_key": "one", "cases": [_raw("a")]}))
    with pytest.raises(ValueError, match="not 'two'"):
        load_dataset("two", path)


def test_duplicate_case_ids_and_unsafe_site_keys_are_rejected(tmp_path) -> None:
    path = tmp_path / "dup.json"
    path.write_text(json.dumps({"cases": [_raw("a"), _raw("a")]}))
    with pytest.raises(ValueError, match="duplicate"):
        load_dataset(None, path)
    for key in ("../x", "a/b", ".hidden", ""):
        with pytest.raises(ValueError):
            dataset_path_for(key)
    with pytest.raises(ValueError):
        load_dataset()


def test_shipped_datasets_load_and_name_their_own_site() -> None:
    for key in ("alpha", "beta", "samplesite"):
        site_key, cases = load_dataset(key)
        assert site_key == key
        assert cases


def _raw(case_id: str) -> dict:
    return {"id": case_id, "family": "f", "turns": ["hi"], "expected_state": "bot"}


# --- scoring -------------------------------------------------------------------------------


def test_expected_unknown_requires_the_bot_to_decline() -> None:
    case = _case(expected_unknown=True)
    assert score_case(case, DECLINE).passed is True
    answered = score_case(case, ANSWER)
    assert answered.passed is False
    assert any(item.startswith("abstain:") for item in answered.failures)
    # A canned or synthesized answer with no citation is still an answer.
    uncited_answer = ObservedReply(state="bot", role="bot", body="Sure.", outcome="exact_answer")
    assert score_case(case, uncited_answer).passed is False


def test_citation_expectations() -> None:
    assert score_case(_case(expect_citations=True), ANSWER).passed is True
    assert score_case(_case(expect_citations=True), DECLINE).passed is False
    assert score_case(_case(expect_citations=False), ANSWER).passed is False
    assert score_case(_case(), ANSWER).passed is True


def test_a_citation_from_another_site_always_fails() -> None:
    foreign = ObservedReply(
        state="bot",
        role="bot",
        body="Yes.",
        outcome="synthesized_answer",
        citation_chunk_ids=(uuid4(),),
        foreign_citation_ids=(uuid4(),),
    )
    verdict = score_case(_case(), foreign)
    assert verdict.passed is False
    assert verdict.foreign_citations == 1
    missing = ObservedReply(
        state="bot",
        role="bot",
        body="Yes.",
        citation_chunk_ids=(uuid4(),),
        unresolved_citation_ids=(uuid4(),),
    )
    assert score_case(_case(), missing).passed is False


def test_outcome_locator_and_latency_budget() -> None:
    case = _case(expected_outcome="synthesized_answer", expected_locator="#a", max_latency_ms=100)
    verdict = score_case(case, ANSWER)
    assert {item.split(":")[0] for item in verdict.failures} == {"locator", "latency"}
    assert verdict.latency_ms == 120
    ok = ObservedReply(
        state="bot",
        role="bot",
        body="Y",
        outcome="synthesized_answer",
        display_locator="#a",
        turn_latencies_ms=(40, 90),
    )
    assert score_case(case, ok).passed is True


def test_summary_groups_by_family() -> None:
    verdicts = [
        score_case(EvalCase("a", "grounded.x", ("q",), "bot", expect_citations=True), ANSWER),
        score_case(EvalCase("b", "grounded.x", ("q",), "bot", expect_citations=True), DECLINE),
        score_case(EvalCase("c", "abstain.y", ("q",), "bot", expected_unknown=True), DECLINE),
    ]
    by_family = {item.family: item for item in summarize(verdicts)}
    assert (by_family["grounded.x"].passed, by_family["grounded.x"].total) == (1, 2)
    assert by_family["grounded.x"].cited == 1
    assert (by_family["abstain.y"].abstained_correctly, by_family["abstain.y"].unknown_cases) == (
        1,
        1,
    )
    assert by_family["grounded.x"].p50_ms == 120
    text = format_summary(summarize(verdicts))
    assert "grounded.x" in text and "abstain.y" in text and "1/1" in text


def test_runner_requires_real_providers_and_a_site(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from app.settings import reset_settings_cache

    reset_settings_cache()
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        require_providers()
    with pytest.raises(SystemExit):
        parse_args([])
    assert parse_args(["--site", "alpha"]).site == "alpha"


def test_probe_conditions() -> None:
    turn = {
        "state": "bot",
        "reply": {"body": "Hello there", "display_locator": "#hi", "system_reason": "canned"},
    }
    assert (
        check_chat({"reason": "canned", "locator": "#hi", "body_contains": ["Hello"]}, turn) == []
    )
    unmet = check_chat({"locator_not_in": ["#hi"], "body_min_length": 99, "state": "queued"}, turn)
    assert len(unmet) == 3
    assert probes_path_for("samplesite").name == "samplesite.json"
    with pytest.raises(ValueError):
        probes_path_for("../x")


# --- a real turn, per site -----------------------------------------------------------------


@pytest.mark.parametrize("site_key", [ALPHA_KEY, BETA_KEY])
async def test_site_dataset_passes_against_its_own_site(migrated_db, site_key) -> None:
    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        site = seeded.alpha if site_key == ALPHA_KEY else seeded.beta
        _key, cases = load_dataset(site_key)
        for case in cases:
            observed, verdict = await run_case(
                session,
                site,
                case,
                parent_origin=HOST_ORIGIN,
                responder=ForeignCitingResponder(match_question=True),
                embedder=TopicEmbedder(),
            )
            assert verdict.passed, (case.id, verdict.failures, observed.body)
            assert observed.foreign_citation_ids == ()
            assert observed.latency_ms is not None


async def test_citation_site_check_names_foreign_and_missing_chunks(migrated_db) -> None:
    from app.services.bot_eval import _citation_sites

    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        gone = uuid4()
        foreign, missing = await _citation_sites(
            session,
            seeded.alpha.id,
            [seeded.alpha_shared.id, seeded.beta_shared.id, gone, seeded.beta_shared.id],
        )
    assert foreign == (seeded.beta_shared.id,)
    assert missing == (gone,)


async def test_a_model_citing_the_other_sites_chunk_is_declined_not_persisted(migrated_db) -> None:
    async with session_maker()() as session:
        seeded = await seed_two_sites(session)
        case = EvalCase(
            id="foreign",
            family="abstain.other_site",
            turns=(BETA_ONLY_QUESTION,),
            expected_state="bot",
            expected_unknown=True,
        )
        observed, verdict = await run_case(
            session,
            seeded.alpha,
            case,
            parent_origin=HOST_ORIGIN,
            responder=ForeignCitingResponder(
                cite_chunk_id=seeded.beta_only.id, cite_snapshot_id=seeded.beta_only.snapshot_id
            ),
            embedder=TopicEmbedder(),
        )
    assert observed.foreign_citation_ids == ()
    assert verdict.passed, verdict.failures
