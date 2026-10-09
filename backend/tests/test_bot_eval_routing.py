"""Routing evals run through the harness with a scripted understanding of each message.

The per-site dataset says what a good reply looks like; here the model is a double, so these
tests prove the runner and scorer agree with the routes the application takes (an off-topic
message stays with the bot and never queues a specialist; a request for a person does queue).
Whether a real model reads each message that way is what the live runner measures.
"""

from app.db import session_maker
from app.services.bot_eval import EvalCase, load_dataset, run_case
from tests.bot_fixtures import seed_brand_articles
from tests.fake_anthropic import ScriptedResponder
from tests.fakes import FakeEmbedder
from tests.ws_helpers import HOST_ORIGIN

REDIRECT = "What would you like to know about screening or compliance?"


async def test_off_topic_dataset_cases_stay_with_the_bot_and_never_queue(migrated_db) -> None:
    _site_key, cases = load_dataset("samplesite")
    off_topic = [item for item in cases if item.family == "routing.off_topic"]
    assert len(off_topic) >= 3
    responder = ScriptedResponder(understand={"intent": "off_topic", "reply": REDIRECT})
    async with session_maker()() as session:
        easy, _bg, _timing, _fcra = await seed_brand_articles(session)
        await session.commit()
        failed: list[str] = []
        for case in off_topic:
            observed, verdict = await run_case(
                session,
                easy,
                case,
                parent_origin=HOST_ORIGIN,
                responder=responder,
                embedder=FakeEmbedder(),
            )
            assert observed.body == REDIRECT
            assert observed.state == "bot"
            if not verdict.passed:
                failed.append(f"{case.id}: {', '.join(verdict.failures)}")
        assert failed == []
        assert responder.stage_calls("draft") == []


async def test_request_for_a_person_queues_and_is_not_an_answer(migrated_db) -> None:
    _site_key, cases = load_dataset("samplesite")
    escalate = [item for item in cases if item.family == "routing.escalate"]
    assert escalate
    responder = ScriptedResponder(
        understand={"intent": "handoff_request", "reply": "Connecting you."}
    )
    async with session_maker()() as session:
        easy, _bg, _timing, _fcra = await seed_brand_articles(session)
        await session.commit()
        for case in escalate:
            observed, verdict = await run_case(
                session,
                easy,
                case,
                parent_origin=HOST_ORIGIN,
                responder=responder,
                embedder=FakeEmbedder(),
            )
            assert observed.state == "queued"
            assert observed.system_reason == "visitor_request"
            assert verdict.passed, verdict.failures


async def test_declining_to_answer_satisfies_an_expected_unknown_case(migrated_db) -> None:
    case = EvalCase(
        id="unknown.pricing",
        family="retrieval.miss",
        turns=("what is your unpublished internal pricing matrix",),
        expected_state="bot",
        expected_unknown=True,
        expect_citations=False,
    )
    responder = ScriptedResponder(
        understand={"intent": "off_topic", "reply": "Not something I cover."}
    )
    async with session_maker()() as session:
        easy, _bg, _timing, _fcra = await seed_brand_articles(session)
        await session.commit()
        observed, verdict = await run_case(
            session,
            easy,
            case,
            parent_origin=HOST_ORIGIN,
            responder=responder,
            embedder=FakeEmbedder(),
        )
    assert observed.abstained
    assert verdict.passed, verdict.failures
