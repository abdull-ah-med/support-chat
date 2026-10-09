"""Tokenisation is Unicode-aware and carries no language, product or site vocabulary."""

import pytest
from sqlalchemy import text

from app.db import session_maker
from app.services import kb_tokens
from app.services.kb_tokens import fold_text, search_tokens, token_key, tokenize, tsquery_text


def test_accents_and_case_fold_to_one_token() -> None:
    assert tokenize("¿Cuánto cuesta el ENVÍO?") == ["cuanto", "cuesta", "el", "envio"]
    assert tokenize("envio") == tokenize("Envío")


def test_non_latin_scripts_are_kept_as_words() -> None:
    assert tokenize("Привет, мир") == ["привет", "мир"]
    assert tokenize("日本語 テスト") == ["日本語", "テスト"]


def test_no_word_is_dropped_or_rewritten_by_domain_vocabulary() -> None:
    # Nothing maps cost -> pricing, how long -> turnaround, or drops "how"/"what".
    assert tokenize("How much does it cost, how long?") == [
        "how",
        "much",
        "does",
        "it",
        "cost",
        "how",
        "long",
    ]


def test_single_ascii_letters_from_contractions_are_not_tokens() -> None:
    assert tokenize("what's the capital of France") == ["what", "the", "capital", "of", "france"]


def test_search_tokens_are_distinct_in_first_seen_order() -> None:
    assert search_tokens("Results results RESULTS and more results") == ["results", "and", "more"]


def test_token_key_ignores_punctuation_case_and_accents() -> None:
    assert token_key("  ¿Cuánto tarda el envío? ") == token_key("cuanto tarda el envio")
    assert token_key("What is FCRA?") == "what is fcra"


def test_prefix_query_matches_word_families_for_longer_tokens_only() -> None:
    assert tsquery_text(["fast", "results"], "|") == "fast | resul:*"
    assert tsquery_text(["a1", "shipping"], "&") == "a1 & shippi:*"


def test_module_holds_no_domain_vocabulary() -> None:
    for name in ("ALIASES", "STOP_WORDS", "WEAK_OVERLAP", "GENERIC_NOISE", "OVERVIEW_HINTS"):
        assert not hasattr(kb_tokens, name), name
    assert not hasattr(kb_tokens, "is_overview_query")


@pytest.mark.parametrize(
    "value",
    [
        "Envío rápido ÑANDÚ",
        "Straße GROSS",
        "ﬁnal ① \uff12\uff14",
        "İstanbul",
        "naïve café",
        "日本語",
    ],
)
async def test_python_fold_equals_the_sql_function_used_by_the_indexes(
    migrated_db, value: str
) -> None:
    async with session_maker()() as session:
        folded = await session.scalar(text("SELECT workspace_search_fold(:v)"), {"v": value})
    assert folded == fold_text(value)
