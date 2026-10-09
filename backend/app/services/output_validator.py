"""Server-side safety gate for model-written text before persistence (neutral to language)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from uuid import UUID

import structlog

from app.llm.safety_markers import OUTPUT_LEAK_MARKERS, OUTPUT_PII_PATTERNS
from app.settings import get_settings

log = structlog.get_logger("output_validator")

_HTML_TAG_RE = re.compile(r"<[A-Za-z]")

ValidationReason = str


@dataclass(frozen=True)
class ValidationOutcome:
    accepted: bool
    reason: ValidationReason
    citations: list[UUID] = field(default_factory=list)


def evaluate(  # noqa: C901
    body: str,
    *,
    allowed_ids: list[UUID],
    cited: list[UUID],
    live_urls: set[str] | None = None,
    citation_urls: list[str] | None = None,
    off_brand_blocklist: list[str] | None = None,
    uncited_ok: bool = False,
    max_chars: int | None = None,
    public_contact_text: str = "",
) -> ValidationOutcome:
    text = (body or "").strip()
    citations = list(cited or [])
    limit = max_chars if max_chars is not None else get_settings().max_bot_answer_chars

    if not text:
        return _reject("empty", citations)
    if len(text) > limit:
        return _reject("over_length", citations)
    if not citations and not uncited_ok:
        return _reject("no_citation", citations)

    allowed = set(allowed_ids or [])
    if citations and any(item not in allowed for item in citations):
        return _reject("cite_off_corpus", citations)

    if live_urls is not None and citation_urls:
        for url in citation_urls:
            if url and url not in live_urls:
                return _reject("cite_off_corpus", citations)

    lowered = text.casefold()
    if any(marker in lowered for marker in OUTPUT_LEAK_MARKERS):
        return _reject("injection_leak", citations)

    for index, pattern in enumerate(OUTPUT_PII_PATTERNS):
        approved = (
            set(pattern.findall(public_contact_text))
            if index >= len(OUTPUT_PII_PATTERNS) - 2
            else set()
        )
        if any(match.group() not in approved for match in pattern.finditer(text)):
            return _reject("pii_leak", citations)

    if _HTML_TAG_RE.search(text):
        return _reject("html_leak", citations)

    blocklist = [item.strip().casefold() for item in (off_brand_blocklist or []) if item.strip()]
    if any(token in lowered for token in blocklist):
        return _reject("brand_leak", citations)

    return ValidationOutcome(accepted=True, reason="accepted", citations=citations)


def _reject(reason: ValidationReason, citations: list[UUID]) -> ValidationOutcome:
    log.info("output_reject", reason=reason)
    return ValidationOutcome(accepted=False, reason=reason, citations=citations)
