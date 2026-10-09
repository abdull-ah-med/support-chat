"""Language-neutral text normalisation shared by retrieval and the database.

Nothing here knows a language, a product, or a site. Tokens are Unicode word runs
folded the same way PostgreSQL folds them (see ``workspace_search_fold``), so a query
written without accents matches stored text with accents and the other way round.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_COMBINING_MARKS_RE = re.compile("[̀-ͯ]")
_MIN_ASCII_TOKEN = 2
# Query tokens at least this long also match longer words that share their leading
# letters (language-neutral stand-in for stemming: result / results / resulting).
PREFIX_MIN_TOKEN = 5
PREFIX_TRIM = 2
PREFIX_MIN_STEM = 4


def normalize_query(value: str) -> str:
    """Stable cache/identity key: NFKC, case-folded, single-spaced."""
    return " ".join(unicodedata.normalize("NFKC", value or "").casefold().split())


def fold_text(value: str) -> str:
    """Mirror of the SQL function ``workspace_search_fold`` (NFKD, no marks, lower)."""
    decomposed = unicodedata.normalize("NFKD", value or "")
    return _COMBINING_MARKS_RE.sub("", decomposed).lower()


def tokenize(value: str) -> list[str]:
    """Folded Unicode word tokens in order. Single ASCII characters carry no signal."""
    return [
        token
        for token in _WORD_RE.findall(fold_text(value))
        if len(token) >= _MIN_ASCII_TOKEN or not token.isascii()
    ]


def search_tokens(value: str) -> list[str]:
    """Distinct tokens, first occurrence order, for building a text-search query."""
    return list(dict.fromkeys(tokenize(value)))


def token_key(value: str) -> str:
    """Punctuation-, case- and accent-insensitive identity of a phrase."""
    return " ".join(_WORD_RE.findall(fold_text(value)))


def tsquery_text(tokens: Sequence[str], joiner: str) -> str:
    """Text for ``to_tsquery('simple', ...)``: tokens joined by ``|`` or ``&``.

    Tokens come from :func:`tokenize`, so they hold only letters and digits and cannot
    carry tsquery operators. Longer tokens match by prefix.
    """
    terms: list[str] = []
    for token in tokens:
        if len(token) >= PREFIX_MIN_TOKEN:
            terms.append(f"{token[: max(PREFIX_MIN_STEM, len(token) - PREFIX_TRIM)]}:*")
        else:
            terms.append(token)
    return f" {joiner} ".join(terms)
