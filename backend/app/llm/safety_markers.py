"""Language- and domain-neutral guards: PII shapes, output leaks, document injection markers."""

from __future__ import annotations

import re

# Applied to every retrieved knowledge document body BEFORE generation (defense in depth).
# Visitor text is never classified with keyword lists: the model judges what it means.
INJECTION_MARKERS = (
    "ignore previous instructions",
    "ignore prior instructions",
    "disregard the system",
    "reveal the system prompt",
    "reveal your prompt",
    "act as admin",
    "act as supportchat admin",
    "developer mode",
    "you are now",
    "jailbreak",
)

# Applied to the reply body only.
OUTPUT_LEAK_MARKERS = (
    "system prompt",
    "reveal your prompt",
    "here are my rules",
    "my instructions are",
    "i was told to",
    "supportchat assistant",
    "<script",
    "javascript:",
    "onerror=",
)

# Applied to the reply body: PII patterns.
OUTPUT_PII_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN
    re.compile(r"\b\d{9}\b"),  # SSN no dashes
    re.compile(r"\b[A-Z]\d{7,9}\b"),  # US driver's license shape
    re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"),  # ISO DOB
    # Mixed letters and digits only: all-letter shapes are commonly business acronyms.
    re.compile(r"\b(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9]{2,3}-?[A-Z0-9]{2,4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),  # email
    re.compile(
        r"(?<!\d)(?:"
        r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)\s*|\d{3}[-.\s])\d{3}[-.\s]?\d{4}"
        r"|\d{10}"
        r")(?!\d)"
    ),  # US phone
)

# The first four OUTPUT_PII_PATTERNS are personal identifiers (SSN, 9 digits, licence number,
# ISO date of birth). Their presence in a visitor message is a structural fact, not an opinion
# about the words around it.
_IDENTIFIER_LABELS = ("ssn", "digits_9", "licence_number", "date_of_birth")

ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")


def contains_injection_marker(text: str) -> bool:
    lowered = (text or "").casefold()
    return any(marker in lowered for marker in INJECTION_MARKERS)


def strip_invisible(text: str) -> str:
    return ZERO_WIDTH_RE.sub("", text or "")


def structured_identifier_kinds(text: str) -> list[str]:
    """Labels of identifier-shaped values in visitor text (empty when there are none)."""
    cleaned = strip_invisible(text or "")
    return [
        label
        for label, pattern in zip(_IDENTIFIER_LABELS, OUTPUT_PII_PATTERNS, strict=False)
        if pattern.search(cleaned)
    ]


IDENTIFIER_MASK = "[REDACTED]"


def mask_identifiers(text: str) -> str:
    """Replace identifier-shaped values (SSN, 9-digit numbers, licence numbers, dates of birth)
    so the stored visitor message never holds them. Contact details are left alone."""
    cleaned = strip_invisible(text or "")
    for pattern in OUTPUT_PII_PATTERNS[: len(_IDENTIFIER_LABELS)]:
        cleaned = pattern.sub(IDENTIFIER_MASK, cleaned)
    return cleaned


def has_identifier_mask(text: str) -> bool:
    return IDENTIFIER_MASK in (text or "")
