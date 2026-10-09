"""Structural validation of grounded drafts against live evidence.

Everything here is language- and domain-neutral: offsets, source text, site membership, digit
sequences and PII shapes. Whether a claim is supported or answers the question is judged by
the assessor model, not by word lists.
"""

# ruff: noqa: RUF001
from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

import structlog

from app.llm.prompts import document_body
from app.llm.safety_markers import OUTPUT_PII_PATTERNS
from app.services import output_validator
from app.services.grounded_response_types import (
    Citation,
    DraftValidation,
    EvidenceUnit,
    ModelDraft,
    TurnContext,
)
from app.services.pii_redactor import redact_evidence
from app.settings import get_settings

log = structlog.get_logger("grounded_response")

# Sentence ends: Latin, CJK and Arabic terminators, or the end of the text.
_SENTENCE_RE = re.compile(r"\S.*?(?:[.!?。！？؟](?=\s|$)|[。！？]|$)", re.S)
_DIGIT_RUN_RE = re.compile(r"\d+")
_GROUPING_RE = re.compile(r"(?<=\d)[,.  ](?=\d{3}(?!\d))")


def sentence_spans(body: str) -> list[re.Match[str]]:
    return list(_SENTENCE_RE.finditer(body))


def _ascii_digits(text: str) -> str:
    return "".join(
        str(unicodedata.decimal(char)) if char.isdecimal() else char for char in text or ""
    )


def digit_runs(text: str) -> set[str]:
    """Digit sequences in any script, thousands grouping removed ("1,000" == "1.000" == "1000")."""
    canonical = _GROUPING_RE.sub("", _ascii_digits(text))
    return set(_DIGIT_RUN_RE.findall(canonical))


def numbers_are_supported(body: str, *sources: str) -> bool:
    """Every digit sequence in the answer occurs in the cited text or the visitor's own words."""
    allowed: set[str] = set()
    for source in sources:
        allowed |= digit_runs(source)
    return digit_runs(body) <= allowed


def _normalize_copy(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _is_disallowed_source_copy(body: str, units: list[EvidenceUnit]) -> bool:
    """A reply that is nothing but pasted passages (two or more) is a list, not an answer."""
    verbatim = {_normalize_copy(unit.answer_verbatim) for unit in units if unit.answer_verbatim}
    verbatim.discard("")
    if not verbatim:
        return False
    fragments = [fragment for fragment in re.split(r"\n\s*\n", body) if fragment.strip()]
    if len(fragments) < 2:
        return False
    return all(_normalize_copy(fragment) in verbatim for fragment in fragments)


def uncovered_sentences(draft: ModelDraft) -> list[str]:
    """Sentences no citation overlaps. Structural only: the assessor decides what they mean."""
    return [
        sentence.group().strip()
        for sentence in sentence_spans(draft.body.strip())
        if re.search(r"\w", sentence.group())
        and not any(
            citation.response_start < sentence.end() and citation.response_end > sentence.start()
            for citation in draft.citations
        )
    ]


def remove_draft_sentences(draft: ModelDraft, selected: list[str]) -> ModelDraft | None:
    """Delete only whole, exact sentences and remap retained citation offsets.

    Selection is untrusted model data: it can remove text, never add claims or
    evidence. Callers must validate and reassess the remaining answer.
    """
    spans = sentence_spans(draft.body)
    removals = {text.strip() for text in selected}
    if not removals or not removals.issubset({span.group().strip() for span in spans}):
        return None
    parts, citations, length = [], [], 0
    for span in spans:
        if span.group().strip() in removals:
            continue
        separator = " " if parts else ""
        offset = length + len(separator) - span.start()
        parts.append(separator + span.group())
        length += len(parts[-1])
        citations.extend(
            replace(
                citation,
                response_start=max(citation.response_start, span.start()) + offset,
                response_end=min(citation.response_end, span.end()) + offset,
            )
            for citation in draft.citations
            if citation.response_start < span.end() and citation.response_end > span.start()
        )
    body = "".join(parts)
    return ModelDraft(body, citations, draft.request_id) if body.strip() else None


def attribute_contact_literals(draft: ModelDraft, units: list[EvidenceUnit]) -> ModelDraft:
    """Attribute omitted contact literals only when they exactly occur in evidence.

    This supports only the literal email/phone, not surrounding claims about
    booking, availability, or callbacks. Full validation and semantic review
    still apply to the answer.
    """
    citations = list(draft.citations)
    for pattern in OUTPUT_PII_PATTERNS[-2:]:
        for match in pattern.finditer(draft.body):
            contact = match.group()
            if any(
                item.response_start <= match.start()
                and item.response_end >= match.end()
                and contact in item.cited_text
                for item in citations
            ):
                continue
            for unit in units:
                source = redact_evidence(document_body(unit))
                start = source.find(contact)
                if start < 0:
                    continue
                citations.append(
                    Citation(
                        unit.id,
                        unit.snapshot_id,
                        match.start(),
                        match.end(),
                        start,
                        start + len(contact),
                        contact,
                        unit.source_title,
                        unit.source_url,
                    )
                )
                break
    return ModelDraft(draft.body, citations, draft.request_id)


def _visitor_words(turn: TurnContext) -> list[str]:
    return [
        turn.visitor_text,
        *(
            message.get("content") or ""
            for message in turn.prior_messages
            if message.get("role") == "user"
        ),
    ]


def validate_draft(  # noqa: C901
    draft: ModelDraft,
    units: list[EvidenceUnit],
    turn: TurnContext,
) -> DraftValidation:
    body = draft.body.strip()
    citations = list(draft.citations)
    if not body:
        return _reject("empty", draft)

    allowed = {unit.id: unit for unit in units}
    cited_parts: list[str] = []
    for citation in citations:
        unit = allowed.get(citation.chunk_id)
        if unit is None:
            return _reject("cite_off_corpus", draft)
        if citation.snapshot_id != unit.snapshot_id:
            return _reject("snapshot_mismatch", draft)
        if citation.response_start < 0 or citation.response_end > len(body):
            return _reject("response_offset", draft)
        if citation.response_start >= citation.response_end:
            return _reject("response_offset", draft)
        # Native documents include the source heading. Both forms must quote actual source text.
        sources = [
            redact_evidence(document_body(unit)),
            redact_evidence(unit.answer_verbatim),
        ]
        if not any(
            0 <= citation.source_start < citation.source_end <= len(source) for source in sources
        ):
            return _reject("source_offset", draft)
        if not any(
            citation.cited_text == source[citation.source_start : citation.source_end]
            for source in sources
        ):
            return _reject("source_text_mismatch", draft)
        if citation.source_url != unit.source_url:
            return _reject("source_metadata_mismatch", draft)
        cited_parts.append(citation.cited_text)

    # Length is bounded by characters only; word counts are not a safe proxy across languages.
    max_chars = get_settings().max_bot_answer_chars
    if len(body) > max_chars:
        return _reject("over_length", draft)
    if _is_disallowed_source_copy(body, units):
        return _reject("source_copy", draft)
    # Any answer about the company needs at least one citation. Which sentences need one,
    # and whether the cited span really supports them, is the assessor's judgement.
    if not citations:
        return _reject("no_citation", draft)

    cited_text = "\n".join(cited_parts)
    # A name such as "247 SampleSite" sits in the title of the page the answer cites; its digits
    # are shown to the model with the passage, so repeating them is not an invented number.
    cited_labels = [
        label
        for chunk_id in dict.fromkeys(citation.chunk_id for citation in citations)
        for label in (allowed[chunk_id].source_title, allowed[chunk_id].source_heading)
    ]
    if not numbers_are_supported(
        body, cited_text, *cited_labels, turn.site_name, *_visitor_words(turn)
    ):
        return _reject("unsupported_numeric", draft)

    safety = output_validator.evaluate(
        body,
        allowed_ids=[unit.id for unit in units],
        cited=[citation.chunk_id for citation in citations],
        citation_urls=[citation.source_url for citation in citations],
        live_urls={unit.source_url for unit in units if unit.source_url},
        off_brand_blocklist=list(turn.off_brand_blocklist),
        max_chars=max_chars,
        public_contact_text=cited_text,
    )
    if not safety.accepted:
        return _reject(safety.reason, draft)

    return DraftValidation(
        accepted=True,
        reason="accepted",
        draft=ModelDraft(body=body, citations=citations, request_id=draft.request_id),
    )


def _reject(reason: str, draft: ModelDraft) -> DraftValidation:
    log.info(
        "grounded_draft_reject",
        reason=reason,
        citation_count=len(draft.citations),
        body_chars=len(draft.body or ""),
    )
    return DraftValidation(accepted=False, reason=reason)
