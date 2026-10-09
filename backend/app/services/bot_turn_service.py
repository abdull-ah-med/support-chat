"""Prepare a reply for one visitor turn: understand it, then route, retrieve once and answer.

ConversationService alone owns generation leases, transitions and persistence.

Turn flow (one model call each unless noted):

    understand_turn ──► routed without knowledge? ──► greeting / boundary / handoff / contact /
         │                                            source / clarifying question  (done)
         ▼
    retrieve once ──► relevant evidence? ──no──► "unknown" reply + gap  (no draft call)
         │
         ├─► canned reply (only in the site's primary language) ──► assess ──reject──┐
         ▼                                                                           │
    draft from documents ◄─────────────────────────────────────────────────────────────┘
         │  <no_answer/> ─► "unknown" reply
         ▼
    structural validation ──fail──► one repair ──fail──► "unknown" reply
         ▼
    assess (skipped for plain, standalone, fully cited answers in the site's language)
         └─reject─► trim or one repair ─► reassess ─reject─► "unknown" reply
"""

import asyncio
import re
import time
from dataclasses import dataclass, replace
from uuid import UUID

import structlog
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from app.chat.outcome_copy import handoff_copy
from app.llm.bot_responder import BotResponder
from app.llm.safety_markers import (
    OUTPUT_PII_PATTERNS,
    has_identifier_mask,
    structured_identifier_kinds,
)
from app.llm.understanding import KNOWLEDGE_INTENTS, UnderstandingInput, same_language
from app.models.conversation import Conversation
from app.models.kb_chunk import KbChunk
from app.models.kb_page import KbPage
from app.models.message import Message
from app.models.site import Site
from app.models.visitor import Visitor
from app.repositories.kb_chunk_repo import live_chunks_query
from app.repositories.message_repo import MessageRepository
from app.services.bot_trace import record_trace
from app.services.canned_bot import canned_reply_decision
from app.services.full_context import prior_provider_messages
from app.services.grounded_response import (
    OFFERED_PERSON_REASONS,
    Citation,
    EvidenceUnit,
    GroundedResponseEngine,
    ModelDraft,
    ResponseDecision,
    ResponseOutcome,
    TurnContext,
    _safe_technical_failure,
    eligible_units,
    is_no_answer,
    unknown_decision,
)
from app.services.grounded_response_types import ProviderStatus
from app.services.grounded_response_validation import (
    attribute_contact_literals,
    remove_draft_sentences,
    uncovered_sentences,
    validate_draft,
)
from app.services.kb_embedder import default_embedder
from app.services.kb_hybrid import HybridKbSearch, KbHits
from app.services.pii_redactor import redact_for_log, redact_for_model
from app.services.rate_ceiling import TurnBudget, bind_turn_budget
from app.services.rate_limit import RateLimitExceeded
from app.services.route_decision import live_snapshots_for_site
from app.services.turn_routing import SiteProfile, route_without_knowledge
from app.settings import get_settings

log = structlog.get_logger("bot_turn")

MAX_TURN_EVIDENCE = 8
MAX_CARRIED_EVIDENCE = 2
_REJECT_REASONS = frozenset({"wrong_subject", "wrong_action", "unsupported_claim", "irrelevant"})
# Decisions that already are a final "I can't answer": nothing left to assess.
_FINAL_REASONS = frozenset({"no_evidence", "repeated_miss", "grounding_reject"})


def without_site_name(query: str, site_name: str, visitor_text: str) -> str:
    """The search query minus the site's own name, unless the visitor wrote it.

    Every search is already limited to the site's content, so the brand only pulls in the
    pages that repeat it (headers, legal text) and away from the passage that answers.
    """
    name = (site_name or "").strip()
    cleaned = query.strip()
    if not name or name.casefold() in (visitor_text or "").casefold():
        return cleaned
    return " ".join(re.sub(re.escape(name), " ", cleaned, flags=re.IGNORECASE).split())


@dataclass
class PreparedBotReply:
    decision: ResponseDecision
    stage_timings: dict[str, int]
    # The understanding's intent label, stored on the conversation. None when the turn failed
    # before it was understood.
    intent: str | None = None


class BotTurnService:
    def __init__(self, session: AsyncSession, responder=None, embedder=None) -> None:
        self._session = session
        self._messages = MessageRepository(session)
        self._responder = responder if responder is not None else BotResponder()
        self._embedder = embedder if embedder is not None else default_embedder()

    async def prepare(
        self,
        conversation: Conversation,
        site: Site,
        visitor_text: str,
    ) -> PreparedBotReply:
        """Run the turn with every Claude call counted against the conversation's ceiling."""
        with bind_turn_budget(TurnBudget(conversation.id)) as budget:
            prepared = await self._prepare(conversation, site, visitor_text)
        if budget.tripped is None:
            return prepared
        # Over the ceiling (or the counter is down): stop spending and answer safely.
        # The visitor-submit limiter fails closed on a Redis outage, so this does too.
        if budget.tripped is RateLimitExceeded:
            decision = ResponseDecision(
                ResponseOutcome.KNOWLEDGE_GAP,
                "rate_ceiling",
                handoff_copy("rate_ceiling", human_enabled=site.human_enabled),
                offer_handoff=site.human_enabled,
                provider_status=ProviderStatus.TECH_FAIL,
            )
        else:
            decision = _safe_technical_failure(
                reason="rate_store_unavailable", human_enabled=site.human_enabled
            )
        return PreparedBotReply(decision, prepared.stage_timings, prepared.intent)

    async def _prepare(
        self,
        conversation: Conversation,
        site: Site,
        visitor_text: str,
    ) -> PreparedBotReply:
        timings: dict[str, int] = {
            "history_load": 0,
            "understand": 0,
            "lexical_retrieve": 0,
            "trigram_retrieve": 0,
            "dense_retrieve": 0,
            "embed": 0,
            "provider": 0,
            "citation_parse": 0,
            "validate": 0,
            "liveness_check": 0,
            "commit": 0,
        }
        profile = SiteProfile.of(site)
        started = time.perf_counter_ns()
        window = await self._messages.list_recent_roles(
            conversation.id,
            {"visitor", "bot", "agent"},
            get_settings().conversation_window_size + 1,
        )
        prior_messages = tuple(prior_provider_messages(window, visitor_text))
        timings["history_load"] = (time.perf_counter_ns() - started) // 1_000_000
        last_citations = self._citations_of(
            next((row for row in reversed(window) if row.role == "bot"), None)
        )
        previous_turn = self._previous_turn(window)

        # The stored message is already masked; a mask (or a raw value on a path that did not
        # mask) means the visitor shared an identifier.
        identifiers = structured_identifier_kinds(visitor_text) or (
            ["masked"] if has_identifier_mask(visitor_text) else []
        )
        if identifiers:
            # Structural only: the value is masked before any model sees it.
            log.info(
                "sensitive_intent",
                conversation_id=str(conversation.id),
                kinds=identifiers,
                preview=redact_for_log(visitor_text),
            )

        inputs = UnderstandingInput(
            site_name=profile.name,
            primary_language=profile.primary_language,
            assistant_scope=profile.assistant_scope,
            knowledge_topics=(
                () if profile.assistant_scope else await self._knowledge_topics(site.id)
            ),
            handoff_topics=profile.handoff_topics,
            human_enabled=profile.human_enabled,
            contact_available=bool(profile.contact_info),
            history=prior_messages,
            latest_message=visitor_text,
            previous_reply_kind=previous_turn.response_reason_code if previous_turn else None,
            previous_reply_was_clarification=bool(
                previous_turn
                and previous_turn.response_outcome == ResponseOutcome.CLARIFICATION.value
            ),
            last_answer_had_sources=bool(last_citations),
            identifiers_redacted=bool(identifiers),
            handoff_pending=conversation.state == "queued",
            previous_reply_offered_person=bool(
                profile.human_enabled
                and previous_turn
                and previous_turn.response_reason_code in OFFERED_PERSON_REASONS
            ),
        )
        started = time.perf_counter_ns()
        try:
            understanding = await self._responder.understand_turn(inputs)
        except Exception as exc:
            # No understanding means no safe route. Say so honestly; never guess a reading.
            record_trace("understand_turn", error_class=type(exc).__name__)
            log.info("understanding_failed", error_class=type(exc).__name__)
            return PreparedBotReply(
                _safe_technical_failure(
                    reason="understanding_failed", human_enabled=profile.human_enabled
                ),
                timings,
            )
        finally:
            timings["understand"] = (time.perf_counter_ns() - started) // 1_000_000

        visitor_words = [
            visitor_text,
            *(row["content"] for row in prior_messages if row.get("role") == "user"),
        ]
        routed = route_without_knowledge(
            understanding,
            profile,
            visitor_words=visitor_words,
            last_citations=last_citations,
            previous_was_clarification=inputs.previous_reply_was_clarification,
            identifiers_found=bool(identifiers),
            handoff_pending=inputs.handoff_pending,
        )
        withdrawn = inputs.handoff_pending and (
            understanding.withdraws_handoff or understanding.intent == "handoff_decline"
        )
        if routed is not None:
            record_trace("routing", route=routed.reason_code, knowledge=False)
            return PreparedBotReply(
                replace(routed, cancel_handoff=withdrawn or routed.cancel_handoff),
                timings,
                understanding.intent,
            )

        turn = TurnContext(
            visitor_text=visitor_text,
            evidence=[],
            site_name=profile.name,
            assistant_scope=profile.assistant_scope,
            primary_language=profile.primary_language,
            prior_messages=prior_messages,
            prior_miss_count=conversation.fallback_count,
            off_brand_blocklist=profile.off_brand_blocklist,
            understanding=understanding,
            human_enabled=profile.human_enabled,
            handoff_pending=inputs.handoff_pending,
            contact_info=profile.contact_info,
        )
        decision = await self._answer_from_knowledge(conversation, site, turn, window, timings)
        decision = await self._with_unknown_copy(turn, decision, timings)
        return PreparedBotReply(
            replace(decision, cancel_handoff=withdrawn), timings, understanding.intent
        )

    async def _answer_from_knowledge(
        self,
        conversation: Conversation,
        site: Site,
        turn: TurnContext,
        window: list[Message],
        timings: dict[str, int],
    ) -> ResponseDecision:
        understanding = turn.understanding
        assert understanding is not None
        # The model sees a masked message, but nothing identifier-shaped may reach search or
        # the embedding provider whatever it echoes back.
        query = redact_for_model(
            without_site_name(understanding.query, turn.site_name, turn.visitor_text)
            or turn.visitor_text
        )
        continuation = understanding.relation == "continuation"

        # Retrieve once. The same hits feed the evidence, the relevance gate and the canned match.
        hits = await self._search(
            site.id, query, timings, overview=understanding.intent == "overview"
        )
        current = self._relevant(self._units_from_hits(hits), hits.best_cosine)
        carried = await self._load_carried_evidence(site.id, window) if continuation else []
        evidence = self._merge_evidence(current, carried)
        record_trace(
            "retrieval",
            query=redact_for_model(query),
            original_query=redact_for_model(turn.visitor_text),
            current_evidence_ids=[unit.id for unit in current],
            carried_evidence_ids=[unit.id for unit in carried],
            evidence=evidence,
        )
        turn = replace(turn, evidence=evidence)

        canned = await self._canned_if_same_language(conversation, site, turn, query, window, hits)
        if canned is not None:
            assessed = await self._assess_answer(turn, canned, timings)
            if assessed is not None:
                return assessed
            record_trace("retrieval", canned_rejected=True)

        decision = await self._engine(timings).respond(turn, timings)
        assessed = await self._assess_answer(turn, decision, timings)
        # Only a canned reply can come back rejected as None; a draft always resolves.
        return (
            assessed if assessed is not None else unknown_decision(turn, reason="grounding_reject")
        )

    async def _with_unknown_copy(
        self, turn: TurnContext, decision: ResponseDecision, timings: dict[str, int]
    ) -> ResponseDecision:
        """Give a "no verified information" decision its model-written, visitor-language text."""
        writer = getattr(self._responder, "write_unknown_reply", None)
        if (
            writer is None
            or decision.outcome is not ResponseOutcome.KNOWLEDGE_GAP
            or decision.reason_code not in _FINAL_REASONS
            or decision.provider_status is ProviderStatus.TECH_FAIL
        ):
            return decision
        started = time.perf_counter_ns()
        try:
            text = await writer(turn)
        except Exception as exc:
            record_trace("unknown_reply", error_class=type(exc).__name__)
            return decision
        finally:
            timings["unknown_reply"] = (time.perf_counter_ns() - started) // 1_000_000
        written = unknown_decision(
            turn, reason=decision.reason_code, request_id=decision.request_id, text=text
        )
        record_trace("unknown_reply", text=text, used=written.body != decision.body)
        return replace(written, repair_used=decision.repair_used)

    # ---- assessment -------------------------------------------------------------------

    @staticmethod
    def is_plain_cited_answer(turn: TurnContext, decision: ResponseDecision) -> bool:
        """A standalone factual answer whose every sentence a citation covers.

        The question was unambiguous and self-contained, nothing depends on earlier turns,
        and no sentence is left for the model to have invented. Contact literals stay in the
        assessed set: a cited address can still be offered for the wrong purpose.
        """
        understanding = turn.understanding
        if (
            understanding is None
            or decision.reason_code is not None
            or decision.outcome is not ResponseOutcome.SYNTHESIZED_ANSWER
            or decision.provider_status is not ProviderStatus.OK
            or not decision.citations
            or understanding.intent not in KNOWLEDGE_INTENTS
            or understanding.relation != "standalone"
            or understanding.ambiguity.strip()
        ):
            return False
        if any(pattern.search(decision.body) for pattern in OUTPUT_PII_PATTERNS[-2:]):
            return False
        return not uncovered_sentences(ModelDraft(decision.body, decision.citations))

    @classmethod
    def assessment_required(cls, turn: TurnContext, decision: ResponseDecision) -> bool:
        """Whether the assessor model must review this answer.

        Skipped only when (a) the answer is a plain, standalone, fully cited answer and (b) the
        visitor wrote in the site's primary language, so no translation can have bent a claim.
        Continuations, ambiguity, partial answers (any uncovered sentence), contact content and
        every canned reply are always assessed. ``answer_assessor_mode="always"`` assesses all.
        """
        if get_settings().answer_assessor_mode == "always":
            return True
        understanding = turn.understanding
        if understanding is None:
            return True
        return not (
            cls.is_plain_cited_answer(turn, decision)
            and same_language(understanding.language, turn.primary_language)
        )

    async def _assess_answer(
        self,
        turn: TurnContext,
        decision: ResponseDecision,
        timings: dict[str, int],
        *,
        allow_edit: bool = True,
    ) -> ResponseDecision | None:
        """Assess a decision. None means a canned reply was rejected (fall back to knowledge)."""
        canned = decision.reason_code == "canned_reply"
        assessor = getattr(self._responder, "assess_answer", None)
        if (
            assessor is None
            or decision.provider_status is ProviderStatus.TECH_FAIL
            or decision.reason_code in _FINAL_REASONS
            or decision.outcome is ResponseOutcome.BOUNDARY
            or not self.assessment_required(turn, decision)
        ):
            return decision
        started = time.perf_counter_ns()
        try:
            assessment = await assessor(turn, decision)
        except Exception as exc:
            record_trace("assess_answer", error_class=type(exc).__name__)
            return self._assessor_unavailable(turn, decision)
        finally:
            timings["assess_answer"] = (
                timings.get("assess_answer", 0) + (time.perf_counter_ns() - started) // 1_000_000
            )
        record_trace(
            "answer_quality", relevance=assessment.status, relevance_reason=assessment.reason
        )
        if assessment.status == "reject" or assessment.reason in _REJECT_REASONS:
            if canned:
                return None
            return await self._after_rejection(turn, decision, assessment, timings, allow_edit)
        if assessment.status == "supported_next_step":
            return replace(
                decision,
                outcome=ResponseOutcome.PARTIAL_ANSWER,
                reason_code="canned_reply" if canned else "supported_next_step",
            )
        if assessment.status == "unsupported_detail":
            return replace(
                decision,
                outcome=ResponseOutcome.PARTIAL_ANSWER
                if decision.citations
                else ResponseOutcome.KNOWLEDGE_GAP,
                reason_code="needs_confirmation",
            )
        if assessment.status == "clarification":
            return replace(decision, outcome=ResponseOutcome.CLARIFICATION, reason_code=None)
        return decision

    async def _after_rejection(
        self,
        turn: TurnContext,
        decision: ResponseDecision,
        assessment,
        timings: dict[str, int],
        allow_edit: bool,
    ) -> ResponseDecision:
        """Reviewer rejected a drafted answer: one edit (trim, else one repair), else say so."""
        if allow_edit:
            trimmed = self._trim_relevance(turn, decision, assessment.remove_sentences)
            if trimmed is not None:
                return await self._reassessed(turn, trimmed, timings)
            if not decision.repair_used:
                repaired = await self._repair_after_assessment(
                    turn, decision, assessment.repair_instruction or assessment.reason, timings
                )
                if repaired is not None:
                    return await self._reassessed(turn, repaired, timings)
        return replace(
            unknown_decision(turn, reason="grounding_reject", request_id=decision.request_id),
            repair_used=decision.repair_used,
        )

    async def _reassessed(
        self, turn: TurnContext, decision: ResponseDecision, timings: dict[str, int]
    ) -> ResponseDecision:
        assessed = await self._assess_answer(turn, decision, timings, allow_edit=False)
        return (
            assessed if assessed is not None else unknown_decision(turn, reason="grounding_reject")
        )

    @classmethod
    def _assessor_unavailable(
        cls, turn: TurnContext, decision: ResponseDecision
    ) -> ResponseDecision:
        """An assessor outage may release only an answer that needed no judgement call.

        Plain, standalone, fully cited answers passed every structural check; everything that
        depends on earlier turns, on ambiguity or on uncovered text stays strict and fails
        honestly instead.
        """
        if decision.reason_code != "canned_reply" and cls.is_plain_cited_answer(turn, decision):
            record_trace("assess_answer", released_unassessed=True)
            return decision
        return _safe_technical_failure(reason="answer_assessment", human_enabled=turn.human_enabled)

    @staticmethod
    def _trim_relevance(
        turn: TurnContext, decision: ResponseDecision, selected: list[str]
    ) -> ResponseDecision | None:
        trimmed = remove_draft_sentences(
            ModelDraft(decision.body, decision.citations, decision.request_id), selected
        )
        if trimmed is None:
            return None
        validation = validate_draft(trimmed, turn.evidence, turn)
        if not validation.accepted or validation.draft is None:
            return None
        record_trace("relevance_edit", removed=selected)
        return replace(
            decision,
            body=validation.draft.body,
            citations=validation.draft.citations,
        )

    async def _repair_after_assessment(
        self, turn: TurnContext, decision: ResponseDecision, failure: str, timings: dict[str, int]
    ) -> ResponseDecision | None:
        """The turn's one repair attempt, spent on reviewer feedback."""
        repair = getattr(self._responder, "repair_grounded_draft", None)
        if repair is None:
            return None
        evidence = eligible_units(turn.evidence)
        started = time.perf_counter_ns()
        record_trace("relevance_repair", attempted=True, reason=failure)
        draft = ModelDraft(decision.body, decision.citations, decision.request_id)
        try:
            async with asyncio.timeout(get_settings().anthropic_timeout):
                repaired = await repair(turn, evidence, draft, failure, stage_timings=timings)
        except Exception:
            repaired = None
        timings["relevance_repair"] = (time.perf_counter_ns() - started) // 1_000_000
        if repaired is None or not repaired.body.strip():
            return None
        if is_no_answer(repaired.body):
            return replace(
                unknown_decision(turn, reason="grounding_reject", request_id=repaired.request_id),
                repair_used=True,
            )
        repaired = attribute_contact_literals(repaired, evidence)
        validation = validate_draft(repaired, evidence, turn)
        record_trace(
            "relevance_repair", accepted=validation.accepted, reason_code=validation.reason
        )
        if not validation.accepted or validation.draft is None:
            return None
        return ResponseDecision(
            ResponseOutcome.SYNTHESIZED_ANSWER,
            None,
            validation.draft.body,
            citations=validation.draft.citations,
            provider_status=ProviderStatus.OK,
            request_id=validation.draft.request_id,
            repair_used=True,
        )

    def _engine(self, timings: dict[str, int]) -> GroundedResponseEngine:
        draft = getattr(self._responder, "generate_grounded_draft", None)
        repair_draft = getattr(self._responder, "repair_grounded_draft", None)

        async def complete(turn, units):
            return await draft(turn, units, stage_timings=timings)

        repair = None
        if repair_draft is not None:

            async def repair(turn, units, previous, failure):
                # The budget starts now, not at engine build: a slow or retried draft must not
                # starve the repair. The outer turn deadline bounds the whole turn.
                budget = get_settings().anthropic_timeout
                if budget <= 0:
                    record_trace("repair", budget_exhausted=True)
                    return None
                async with asyncio.timeout(budget):
                    return await repair_draft(turn, units, previous, failure, stage_timings=timings)

        return GroundedResponseEngine(complete=complete if draft else None, repair=repair)

    # ---- canned replies ---------------------------------------------------------------

    async def _canned_if_same_language(
        self,
        conversation: Conversation,
        site: Site,
        turn: TurnContext,
        query: str,
        window: list[Message],
        hits: KbHits,
    ) -> ResponseDecision | None:
        """Canned replies are staff-approved copy in the site's language: send only to a visitor
        writing that language, otherwise let the knowledge path answer in theirs."""
        understanding = turn.understanding
        assert understanding is not None
        if not same_language(understanding.language, turn.primary_language):
            record_trace("retrieval", canned_skipped="language", language=understanding.language)
            return None
        visitor = await self._session.get(Visitor, conversation.visitor_id)
        previous = next((row for row in reversed(window) if row.role == "bot"), None)
        decision = await canned_reply_decision(
            self._session,
            self._embedder,
            site.id,
            query,
            visitor,
            previous_locator=previous.display_locator if previous is not None else None,
            kb_hits=hits,
        )
        if decision is None:
            record_trace("retrieval", canned_win=False)
            return None
        record_trace(
            "retrieval",
            canned_win=True,
            canned_shortcut=decision.display_locator,
            query=redact_for_model(query),
        )
        return decision

    # ---- retrieval --------------------------------------------------------------------

    async def _search(
        self, site_id: UUID, query: str, timings: dict[str, int], *, overview: bool
    ) -> KbHits:
        if not await live_snapshots_for_site(self._session, site_id):
            return KbHits()
        search = HybridKbSearch(self._session)
        return await search.search_with_deferred_embed(
            self._embedder, site_id, query, stage_timings=timings, overview=overview
        )

    @staticmethod
    def _units_from_hits(hits: list) -> list[EvidenceUnit]:
        return [
            EvidenceUnit(
                id=hit.id,
                canonical_question=hit.canonical_question,
                aliases=tuple(hit.aliases or ()),
                topic_label=hit.topic_label or hit.heading,
                answer_verbatim=hit.answer_verbatim or hit.body,
                source_title=hit.title,
                source_url=hit.url,
                snapshot_id=hit.snapshot_id,
                risk_class=hit.risk_class,
                answer_mode=hit.answer_mode,
                enabled=hit.enabled,
                live=True,
                source_heading=hit.heading,
                relevance=getattr(hit, "cosine", None),
            )
            for hit in hits
        ]

    @staticmethod
    def _relevant(
        units: list[EvidenceUnit], best_cosine: float | None = None
    ) -> list[EvidenceUnit]:
        """Drop retrieval noise: keep hits only when the best similarity clears the threshold.

        ``best_cosine`` is the retriever's best embedding similarity over all candidates; when
        it is absent the hits' own scores are used. If nothing could be scored (lexical-only
        hits, embedder down) the threshold has nothing to judge and the hits stay.
        """
        scores = [unit.relevance for unit in units if unit.relevance is not None]
        best = best_cosine if best_cosine is not None else (max(scores) if scores else None)
        if units and best is not None and best < get_settings().evidence_min_cosine:
            record_trace("retrieval", below_relevance_threshold=best)
            return []
        return units

    async def _knowledge_topics(self, site_id: UUID) -> tuple[str, ...]:
        """Largest live pages by passage count: a cheap sample of what the site's knowledge is about."""
        result = await self._session.execute(
            live_chunks_query(site_id)
            .with_only_columns(KbPage.title, func.count(KbChunk.id), maintain_column_froms=True)
            .group_by(KbPage.id, KbPage.title)
            .order_by(func.count(KbChunk.id).desc(), KbPage.title)
            .limit(get_settings().understanding_topic_limit)
        )
        return tuple(title for title, _count in result.all() if title)

    async def _load_carried_evidence(
        self,
        site_id: UUID,
        window: list[Message],
    ) -> list[EvidenceUnit]:
        previous = next(
            (row for row in reversed(window) if row.role == "bot" and row.citations),
            None,
        )
        if previous is None:
            return []

        citations = [citation for citation in previous.citations if citation.chunk_id is not None]
        chunk_ids = list(dict.fromkeys(citation.chunk_id for citation in citations))[
            :MAX_CARRIED_EVIDENCE
        ]
        if not chunk_ids:
            return []
        result = await self._session.execute(
            live_chunks_query(site_id).where(KbChunk.id.in_(chunk_ids))
        )
        by_id = {chunk.id: (chunk, page) for chunk, page in result.all()}
        metadata = {citation.chunk_id: citation for citation in citations}
        carried: list[EvidenceUnit] = []
        for chunk_id in chunk_ids:
            row = by_id.get(chunk_id)
            citation = metadata.get(chunk_id)
            if row is None or citation is None:
                continue
            chunk, page = row
            carried.append(
                EvidenceUnit(
                    id=chunk.id,
                    canonical_question=chunk.canonical_question,
                    aliases=tuple(chunk.aliases or ()),
                    topic_label=chunk.topic_label or chunk.heading,
                    answer_verbatim=chunk.answer_verbatim or chunk.body,
                    source_title=citation.source_title or page.title,
                    source_url=citation.source_url or page.public_url,
                    snapshot_id=chunk.snapshot_id,
                    risk_class=chunk.risk_class,
                    answer_mode=chunk.answer_mode,
                    enabled=chunk.enabled,
                    live=True,
                    source_heading=chunk.heading,
                )
            )
        return carried

    @staticmethod
    def _merge_evidence(
        current: list[EvidenceUnit],
        carried: list[EvidenceUnit],
    ) -> list[EvidenceUnit]:
        """Reserve a small continuity budget while keeping current-topic evidence last."""
        current_ids = {unit.id for unit in current}
        continuity = [unit for unit in carried if unit.id not in current_ids][:MAX_CARRIED_EVIDENCE]
        current_budget = MAX_TURN_EVIDENCE - len(continuity)
        return [*continuity, *current[:current_budget]]

    @staticmethod
    def _previous_turn(window: list[Message]) -> Message | None:
        """The assistant reply right before the visitor message being answered, if any."""
        rows = list(window)
        if rows and rows[-1].role == "visitor":
            rows = rows[:-1]
        return rows[-1] if rows and rows[-1].role == "bot" else None

    @staticmethod
    def _citations_of(row: Message | None) -> list[Citation]:
        if row is None:
            return []
        return [
            Citation(
                c.chunk_id,
                c.snapshot_id,
                c.response_start,
                c.response_end,
                c.source_start,
                c.source_end,
                c.cited_text,
                c.source_title,
                c.source_url,
            )
            for c in row.citations
        ]
