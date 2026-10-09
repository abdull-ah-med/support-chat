"""Canned replies are matched on what they answer (aliases and body), never on their slug."""

import uuid

import pytest

from app.db import session_maker
from app.models.canned_reply import CannedReply
from app.services import canned_bot
from app.services.canned_bot import (
    CannedHit,
    canned_embed_text,
    canned_reply_decision,
    embed_replies,
    phrase_coverage,
    pick_canned_winner,
    search_canned,
)
from app.services.kb_hybrid import HybridKbSearch, KbHits
from tests.bot_fixtures import insert_site
from tests.fakes import MappedEmbedder

EMBEDDER = MappedEmbedder(
    {
        "turnaround": 1,
        "how long": 1,
        "speedy": 1,
        "price": 2,
        "quote": 2,
        "cuesta": 3,
        "yes": 4,
    }
)


async def _reply(
    session, site_id, shortcut: str, body: str, aliases=(), follows_id=None, hands_off=False
) -> CannedReply:
    row = CannedReply(
        site_id=site_id,
        shortcut=shortcut,
        body=body,
        enabled=True,
        aliases=list(aliases),
        bot_eligible=True,
        follows_id=follows_id,
        hands_off=hands_off,
    )
    session.add(row)
    await session.flush()
    return row


async def _decide(session, site, text, *, previous=None, kb_hits=None):
    return await canned_reply_decision(
        session, EMBEDDER, site.id, text, None, previous, kb_hits=kb_hits
    )


def _hit(shortcut: str, cosine: float | None, coverage: float = 0.0) -> CannedHit:
    return CannedHit(
        id=uuid.uuid4(),
        shortcut=shortcut,
        body=f"body of {shortcut}",
        rrf=0.03,
        cosine=cosine,
        lexical=coverage > 0,
        phrase_coverage=coverage,
    )


def test_embedding_text_is_the_trigger_phrases_then_the_body_never_the_slug() -> None:
    assert canned_embed_text(["a b", "c"], "Body") == "a b | c\nBody"
    assert canned_embed_text([], "Body") == "Body"


def test_a_trigger_phrase_must_be_said_in_full_and_dominate_the_message() -> None:
    aliases = ["turnaround time", "how long"]
    assert phrase_coverage("Turnaround time?", aliases) == 1.0
    assert phrase_coverage("what is the turnaround time", aliases) == pytest.approx(2 / 5)
    assert phrase_coverage("turnaround", aliases) == 0.0, "half a phrase is not the phrase"
    assert phrase_coverage("¿Cuánto cuesta el envío?", ["cuanto cuesta el envio"]) == 1.0
    assert phrase_coverage("", aliases) == 0.0


def test_pick_prefers_the_phrase_the_visitor_said_over_a_slightly_closer_vector() -> None:
    said = _hit("said", cosine=0.50, coverage=1.0)
    closer = _hit("closer", cosine=0.62)
    assert pick_canned_winner([closer, said], best_kb_cosine=0.99) is said


def test_pick_needs_a_clear_margin_between_replies_on_meaning_alone() -> None:
    first, second = _hit("a", 0.70), _hit("b", 0.66)
    assert pick_canned_winner([first, second], best_kb_cosine=None) is None
    clear = _hit("c", 0.55)
    assert pick_canned_winner([first, clear], best_kb_cosine=None) is first


def test_pick_ignores_weak_meaning_and_replies_the_knowledge_base_answers_better() -> None:
    assert pick_canned_winner([_hit("weak", 0.40)], best_kb_cosine=None) is None
    strong = _hit("strong", 0.70)
    assert pick_canned_winner([strong], best_kb_cosine=0.70) is None
    assert pick_canned_winner([strong], best_kb_cosine=0.69) is strong
    assert pick_canned_winner([strong], best_kb_cosine=None) is strong


def test_two_replies_the_message_names_equally_are_left_to_meaning() -> None:
    one = _hit("one", cosine=0.60, coverage=1.0)
    two = _hit("two", cosine=0.58, coverage=1.0)
    assert pick_canned_winner([one, two], best_kb_cosine=None) is None
    assert (
        pick_canned_winner([_hit("x", None, 1.0), _hit("y", None, 1.0)], best_kb_cosine=None)
        is None
    )


def test_slug_matching_helpers_are_gone() -> None:
    for name in (
        "lexical_shortcut_match",
        "decisive_lexical_hit",
        "lexical_specificity",
        "_key_tokens",
        "CANNED_QUERY_TOPICS",
        "ANSWER_WORDS",
        "canned_embedder",
    ):
        assert not hasattr(canned_bot, name), name


async def test_a_reply_is_not_found_by_its_slug(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "shop", "Shop")
        # Slug says pricing; the reply answers something else entirely.
        row = await _reply(session, site.id, "pricing_overview", "Our office is open weekdays.")
        await embed_replies([row], EMBEDDER)
        await session.commit()
        assert await _decide(session, site, "pricing overview") is None
        assert await _decide(session, site, "pricing") is None


async def test_a_reply_is_found_by_its_alias_phrase_without_any_embedding(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "shop", "Shop")
        await _reply(
            session,
            site.id,
            "x9",
            "Results arrive in two business days.",
            aliases=["turnaround time", "how long do results take"],
        )
        await session.commit()
        # No vectors stored (as right after migration 58): the phrase alone decides.
        decision = await _decide(session, site, "Turnaround time?")
    assert decision is not None
    assert decision.body == "Results arrive in two business days."
    assert decision.display_locator == "#x9"


async def test_a_reply_is_found_by_meaning_when_no_word_is_shared(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "shop", "Shop")
        speedy = await _reply(
            session, site.id, "r1", "Delivery takes two days.", aliases=["speedy delivery"]
        )
        other = await _reply(session, site.id, "r2", "Call us for a quote.", aliases=["quote"])
        await embed_replies([speedy, other], EMBEDDER)
        await session.commit()
        decision = await _decide(session, site, "how long until it gets here")
    assert decision is not None and decision.display_locator == "#r1"


async def test_accented_alias_matches_unaccented_message(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        await _reply(
            session,
            site.id,
            "envio",
            "El envío cuesta cinco euros.",
            aliases=["¿cuánto cuesta el envío?"],
        )
        await session.commit()
        decision = await _decide(session, site, "cuanto cuesta el envio")
    assert decision is not None and decision.body == "El envío cuesta cinco euros."


async def test_a_reply_must_beat_the_knowledge_base_it_is_compared_with(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "shop", "Shop")
        row = await _reply(session, site.id, "r1", "Delivery in two days.", aliases=["speedy"])
        await embed_replies([row], EMBEDDER)
        await session.commit()
        text = "speedy please"
        assert await _decide(session, site, text, kb_hits=KbHits(best_cosine=0.2)) is not None
        assert await _decide(session, site, text, kb_hits=KbHits(best_cosine=1.0)) is None


async def test_supplied_kb_hits_replace_a_second_knowledge_search(migrated_db, monkeypatch) -> None:
    searches = 0
    original = HybridKbSearch.search_with_deferred_embed

    async def counting(self, *args, **kwargs):
        nonlocal searches
        searches += 1
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(HybridKbSearch, "search_with_deferred_embed", counting)
    async with session_maker()() as session:
        site = await insert_site(session, "shop", "Shop")
        row = await _reply(session, site.id, "r1", "Delivery in two days.", aliases=["speedy"])
        await embed_replies([row], EMBEDDER)
        # A live knowledge base exists, so an unsupplied comparison would search it.
        from tests.kb_corpus import ChunkSpec, make_corpus

        await make_corpus(session, site, [[ChunkSpec(answer="Plain page text.")]])
        await session.commit()
        await _decide(session, site, "speedy please", kb_hits=[])
        assert searches == 0
        await _decide(session, site, "speedy please")
        assert searches == 1


async def test_follow_up_replies_only_answer_right_after_their_parent(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "shop", "Shop")
        parent = await _reply(session, site.id, "ask_have_account", "Do you have an account?")
        await _reply(
            session,
            site.id,
            "have_account_yes",
            "Great, sign in from the top right.",
            aliases=["yes"],
            follows_id=parent.id,
        )
        await session.commit()
        assert await _decide(session, site, "yes") is None
        assert await _decide(session, site, "yes", previous="#something_else") is None
        decision = await _decide(session, site, "yes", previous="#ask_have_account")
    assert decision is not None and decision.display_locator == "#have_account_yes"


async def test_a_hands_off_reply_asks_for_a_specialist(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "shop", "Shop")
        await _reply(
            session,
            site.id,
            "review",
            "Our team will review this.",
            aliases=["legal question"],
            hands_off=True,
        )
        await session.commit()
        decision = await _decide(session, site, "legal question")
    assert decision is not None and decision.handoff_reason == "individual_case"


async def test_canned_search_never_crosses_sites_and_never_uses_the_general_library(
    migrated_db,
) -> None:
    async with session_maker()() as session:
        alpha = await insert_site(session, "alpha", "Alpha")
        beta = await insert_site(session, "beta", "Beta")
        own = await _reply(session, alpha.id, "own", "Alpha answer.", aliases=["speedy delivery"])
        theirs = await _reply(
            session, beta.id, "theirs", "Beta answer.", aliases=["speedy delivery"]
        )
        general = await _reply(session, None, "general", "Shared answer.", aliases=["quote"])
        await embed_replies([own, theirs, general], EMBEDDER)
        await session.commit()
        for query in ("speedy delivery", "how long does it take"):
            hits = await search_canned(session, EMBEDDER, alpha.id, query)
            assert theirs.id not in {hit.id for hit in hits}, query
            assert own.id in {hit.id for hit in hits}, query
            assert general.id not in {hit.id for hit in hits}, query
        assert await _decide(session, alpha, "quote") is None
        decision = await _decide(session, alpha, "speedy delivery")
    assert decision is not None and decision.body == "Alpha answer."
