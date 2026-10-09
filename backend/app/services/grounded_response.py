"""Typed, evidence-first decisions for visitor-facing bot replies.

Claude authors every answer; deterministic code owns structural validation and the places where
the bot must stay silent. When the documents do not answer, nothing is drafted or guessed: the
turn takes the "unknown" path, which says so honestly in the visitor's language.
"""

from __future__ import annotations

import re
import time
from dataclasses import replace

import structlog

from app.chat.outcome_copy import (
    INSUFFICIENT_HUMAN,
    INSUFFICIENT_SOLO,
    TECH_FAIL_HUMAN,
    TECH_FAIL_SOLO,
    contact_block,
)
from app.llm.prompts import NO_ANSWER
from app.services import output_validator
from app.services.bot_trace import record_trace
from app.services.grounded_response_types import (
    Citation as Citation,
)
from app.services.grounded_response_types import (
    EvidenceUnit as EvidenceUnit,
)
from app.services.grounded_response_types import (
    ModelDraft as ModelDraft,
)
from app.services.grounded_response_types import (
    Provider,
    ProviderStatus,
    RepairProvider,
    ResponseDecision,
    ResponseOutcome,
    TurnContext,
)
from app.services.grounded_response_validation import (
    attribute_contact_literals,
    numbers_are_supported,
    validate_draft,
)

log = structlog.get_logger("grounded_response")

# Reply kinds that end with an offer of a person (when one can join): a "yes" next accepts it.
OFFERED_PERSON_REASONS = frozenset(
    {"no_evidence", "repeated_miss", "grounding_reject", "tech_fail", "urgent_safety"}
)

# Model-written text that carries no company facts may not carry links either.
_LINK_RE = re.compile(r"(?i)\b(?:https?://|www\.)\S+")


def is_no_answer(body: str) -> bool:
    return NO_ANSWER in (body or "")


def eligible_units(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    return [
        unit for unit in units if unit.enabled and unit.live and unit.answer_mode != "human_only"
    ]


def checked_reply(
    text: str | None,
    *,
    visitor_words: list[str] | tuple[str, ...] = (),
    off_brand_blocklist: list[str] | tuple[str, ...] = (),
    site_labels: list[str] | tuple[str, ...] = (),
) -> str | None:
    """Return model-written conversational text only if it is safe to show, else None.

    Such text may state no company facts, so it may contain no link, contact detail, personal
    identifier or number the visitor did not write themselves. The site's own name and scope
    (``site_labels``) are configuration, not invented facts, so digits in them are allowed.
    Language-neutral by construction.
    """
    body = (text or "").strip()
    if not body or is_no_answer(body) or _LINK_RE.search(body):
        return None
    if not numbers_are_supported(body, *visitor_words, *site_labels):
        return None
    outcome = output_validator.evaluate(
        body,
        allowed_ids=[],
        cited=[],
        uncited_ok=True,
        off_brand_blocklist=list(off_brand_blocklist),
    )
    return body if outcome.accepted else None


def source_followup_decision(
    citations: list[Citation], lead_in: str | None = None
) -> ResponseDecision:
    """Re-show the source of the previous answer: its URL and one complete evidence passage."""
    first = citations[0]
    prefix = (f"{lead_in.strip()}\n\n" if lead_in and lead_in.strip() else "") + (
        f"{first.source_url}\n\n"
    )
    # Preserve one complete evidence passage, never clip its qualifiers.
    body = prefix + first.cited_text
    verified = [replace(first, response_start=len(prefix), response_end=len(body))]
    return ResponseDecision(
        ResponseOutcome.SYNTHESIZED_ANSWER,
        "source_followup",
        body,
        citations=verified,
    )


def unknown_decision(
    turn: TurnContext,
    *,
    reason: str | None = None,
    request_id: str | None = None,
    text: str | None = None,
) -> ResponseDecision:
    """The honest "no verified information" reply, with a route to a person or the contact route.

    ``text`` is the model-written unknown reply (visitor language, no facts), added by the turn
    service after the decision is known. Until then (and if that text is unusable) the neutral
    fixed line stands in. The configured contact route is appended verbatim when no person can
    join the chat.
    """
    checked = (
        checked_reply(
            text,
            visitor_words=_visitor_words(turn),
            off_brand_blocklist=turn.off_brand_blocklist,
            site_labels=(turn.site_name, turn.assistant_scope),
        )
        if text
        else None
    )
    body = checked or (INSUFFICIENT_HUMAN if turn.human_enabled else INSUFFICIENT_SOLO)
    if not turn.human_enabled:
        contacts = contact_block(turn.contact_info)
        if contacts:
            body = f"{body}\n\n{contacts}"
    repeated = turn.prior_miss_count > 0
    return ResponseDecision(
        ResponseOutcome.KNOWLEDGE_GAP,
        reason or ("repeated_miss" if repeated else "no_evidence"),
        body,
        offer_handoff=turn.human_enabled and not turn.handoff_pending,
        provider_status=ProviderStatus.OK,
        request_id=request_id,
    )


def _visitor_words(turn: TurnContext) -> list[str]:
    return [
        turn.visitor_text,
        *(
            message.get("content") or ""
            for message in turn.prior_messages
            if message.get("role") == "user"
        ),
    ]


def _safe_technical_failure(
    reason: str = "tech_fail", *, request_id: str | None = None, human_enabled: bool = True
) -> ResponseDecision:
    log.info("grounded_tech_fail", reason=reason)
    return ResponseDecision(
        ResponseOutcome.KNOWLEDGE_GAP,
        "tech_fail",
        TECH_FAIL_HUMAN if human_enabled else TECH_FAIL_SOLO,
        offer_handoff=human_enabled,
        provider_status=ProviderStatus.TECH_FAIL,
        request_id=request_id,
    )


def _elapsed_ms(started_ns: int) -> int:
    return (time.perf_counter_ns() - started_ns) // 1_000_000


class GroundedResponseEngine:
    """Draft from evidence, validate structurally, repair at most once, else say "unknown"."""

    def __init__(
        self, complete: Provider | None = None, repair: RepairProvider | None = None
    ) -> None:
        self._complete = complete
        self._repair = repair

    async def respond(  # noqa: C901
        self,
        turn: TurnContext,
        stage_timings: dict[str, int] | None = None,
        *,
        allow_repair: bool = True,
    ) -> ResponseDecision:
        timings = stage_timings if stage_timings is not None else {}
        evidence = eligible_units(turn.evidence)
        if not evidence:
            return unknown_decision(turn)
        if self._complete is None:
            return _safe_technical_failure(
                reason="provider_unavailable", human_enabled=turn.human_enabled
            )
        try:
            draft = await self._complete(turn, evidence)
        except Exception:
            return _safe_technical_failure(
                reason="provider_exception", human_enabled=turn.human_enabled
            )
        if draft is None or not draft.body.strip():
            return _safe_technical_failure(reason="empty_draft", human_enabled=turn.human_enabled)
        if is_no_answer(draft.body):
            record_trace("draft", no_answer=True, body=draft.body[:600])
            return unknown_decision(turn, request_id=draft.request_id)

        draft = attribute_contact_literals(draft, evidence)
        started = time.perf_counter_ns()
        record_trace("draft", draft=draft)
        validation = validate_draft(draft, evidence, turn)
        timings["validate"] = _elapsed_ms(started)
        record_trace(
            "validation",
            accepted=validation.accepted,
            reason=validation.reason,
            validated_draft=validation.draft,
        )
        repair_used = False
        if not validation.accepted and allow_repair and self._repair is not None:
            repair_used = True
            started = time.perf_counter_ns()
            try:
                repaired = await self._repair(turn, evidence, draft, validation.reason)
            except Exception:
                repaired = None
            timings["repair"] = _elapsed_ms(started)
            if repaired is not None and is_no_answer(repaired.body):
                return unknown_decision(
                    turn, reason="grounding_reject", request_id=repaired.request_id
                )
            if repaired is not None and repaired.body.strip():
                draft = attribute_contact_literals(repaired, evidence)
                started = time.perf_counter_ns()
                validation = validate_draft(draft, evidence, turn)
                timings["validate"] += _elapsed_ms(started)
            record_trace(
                "repair",
                failure=validation.reason,
                accepted=validation.accepted,
                repaired_draft=repaired,
            )
        if not validation.accepted or validation.draft is None:
            decision = unknown_decision(
                turn, reason="grounding_reject", request_id=draft.request_id
            )
            return replace(decision, repair_used=repair_used)
        return ResponseDecision(
            ResponseOutcome.SYNTHESIZED_ANSWER,
            None,
            validation.draft.body,
            citations=validation.draft.citations,
            provider_status=ProviderStatus.OK,
            request_id=validation.draft.request_id,
            repair_used=repair_used,
        )
