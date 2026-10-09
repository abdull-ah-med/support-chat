from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from app.llm.understanding import TurnUnderstanding


class ResponseOutcome(StrEnum):
    EXACT_ANSWER = "exact_answer"  # Deprecated: no longer produced by respond().
    SYNTHESIZED_ANSWER = "synthesized_answer"
    CLARIFICATION = "clarification"
    PARTIAL_ANSWER = "partial_answer"
    KNOWLEDGE_GAP = "knowledge_gap"
    BOUNDARY = "boundary"


class ProviderStatus(StrEnum):
    NOT_USED = "not_used"
    OK = "ok"
    TECH_FAIL = "tech_fail"


@dataclass(frozen=True)
class EvidenceUnit:
    id: UUID
    canonical_question: str | None
    aliases: tuple[str, ...]
    topic_label: str
    answer_verbatim: str
    source_title: str
    source_url: str
    snapshot_id: UUID | None = None
    risk_class: str = "general"
    answer_mode: str = "paraphrase_allowed"
    enabled: bool = True
    live: bool = True
    source_heading: str = ""
    # Dense similarity to the search query (0-1) when the retriever scored it, else None.
    relevance: float | None = None


@dataclass(frozen=True)
class Citation:
    chunk_id: UUID
    snapshot_id: UUID | None
    response_start: int
    response_end: int
    source_start: int
    source_end: int
    cited_text: str
    source_title: str
    source_url: str


@dataclass(frozen=True)
class TurnContext:
    visitor_text: str
    evidence: list[EvidenceUnit]
    site_name: str = ""
    assistant_scope: str = ""
    primary_language: str = "en"
    prior_messages: tuple[dict[str, str], ...] = ()
    # Misses already recorded on this conversation (fallback_count at the start of the turn).
    prior_miss_count: int = 0
    off_brand_blocklist: tuple[str, ...] = ()
    understanding: TurnUnderstanding | None = None
    human_enabled: bool = True
    # A person was already asked to join this chat and has not yet.
    handoff_pending: bool = False
    # Configured contact route (site.contact_info); shown verbatim, never composed by a model.
    contact_info: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResponseDecision:
    outcome: ResponseOutcome | None
    reason_code: str | None
    body: str
    citations: list[Citation] = field(default_factory=list)
    offer_handoff: bool = False
    provider_status: ProviderStatus = ProviderStatus.NOT_USED
    request_id: str | None = None
    display_locator: str | None = None
    # When set, a specialist is asked to take the chat right after this reply is stored.
    handoff_reason: str | None = None
    # The one repair attempt a turn is allowed has been spent on this answer.
    repair_used: bool = False
    # The visitor withdrew a pending request for a person: stand the handoff down.
    cancel_handoff: bool = False


@dataclass(frozen=True)
class ModelDraft:
    body: str
    citations: list[Citation]
    request_id: str | None = None


@dataclass(frozen=True)
class DraftValidation:
    accepted: bool
    reason: str
    draft: ModelDraft | None = None


Provider = Callable[[TurnContext, list[EvidenceUnit]], Awaitable[ModelDraft | None]]
RepairProvider = Callable[
    [TurnContext, list[EvidenceUnit], ModelDraft, str], Awaitable[ModelDraft | None]
]
