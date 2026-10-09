"""Turn routing driven by the model's understanding of the message.

Pure functions: given what the understanding call returned and the site's configuration, decide
which turns need no knowledge lookup at all (greetings, boundaries, handoffs, contact route,
source re-display, clarifying question) and build their decisions. Everything else is a question
for the knowledge pipeline. No word lists: the language and the meaning are the model's.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.chat.outcome_copy import KEEP_HELPING_LINE, OUT_OF_SCOPE, contact_block
from app.llm.understanding import KNOWLEDGE_INTENTS, TurnUnderstanding
from app.services.grounded_response import (
    Citation,
    checked_reply,
    source_followup_decision,
)
from app.services.grounded_response_types import ProviderStatus, ResponseDecision, ResponseOutcome

# Intents answered by the model's own visitor-language reply and nothing else.
_CONVERSATIONAL_REASONS = {
    "smalltalk": "smalltalk",
    "thanks_or_goodbye": "thanks_or_goodbye",
    "handoff_decline": "handoff_declined",
    "conversation_recap": "conversation_recap",
}
_BOUNDARY_REASONS = {
    "off_topic": "off_topic",
    "abuse": "abuse",
    "prompt_injection": "prompt_injection",
}
# A configured handoff topic never turns an insult or an instruction override into a handoff.
_NEVER_HANDOFF = frozenset({"abuse", "prompt_injection"})
# Neutral last-resort lines, used only when the model's own text is unusable.
URGENT_SAFETY_LINE = (
    "If someone is in danger, please contact your local emergency services right away."
)
IDENTIFIERS_LINE = "Please don't share personal identifiers in chat. How else can I help?"
PENDING_LINE = (
    "A specialist has already been asked to join. You can keep asking me questions meanwhile."
)

# Why a person is asked to take the chat: stored on the handoff and the conversation.
_HANDOFF_ESCALATIONS = {
    "handoff_request": "visitor_request",
    "personal_case": "individual_case",
}


@dataclass(frozen=True)
class SiteProfile:
    """The per-site configuration the router reads. Nothing about a brand or a domain."""

    name: str
    assistant_scope: str
    handoff_topics: tuple[str, ...]
    human_enabled: bool
    contact_info: tuple[str, ...]
    primary_language: str
    off_brand_blocklist: tuple[str, ...]

    @classmethod
    def of(cls, site: object) -> SiteProfile:
        return cls(
            name=str(getattr(site, "name", "") or ""),
            assistant_scope=str(getattr(site, "assistant_scope", "") or "").strip(),
            handoff_topics=tuple(
                str(topic).strip()
                for topic in (getattr(site, "handoff_topics", None) or ())
                if str(topic).strip()
            ),
            human_enabled=bool(getattr(site, "human_enabled", True)),
            contact_info=tuple(
                str(item).strip()
                for item in (getattr(site, "contact_info", None) or ())
                if str(item).strip()
            ),
            primary_language=str(getattr(site, "primary_language", "") or "en"),
            off_brand_blocklist=tuple(getattr(site, "off_brand_blocklist", None) or ()),
        )


def wants_person(understanding: TurnUnderstanding, profile: SiteProfile | None = None) -> bool:
    """A person must take this: asked for, a personal case, or a configured handoff topic.

    A handoff topic only counts when the site configured some: the model cannot match a topic
    that does not exist.
    """
    if understanding.intent in _NEVER_HANDOFF or understanding.intent == "urgent_safety":
        return False
    topic = understanding.matches_handoff_topic and (
        profile is None or bool(profile.handoff_topics)
    )
    return understanding.intent in _HANDOFF_ESCALATIONS or topic


def escalation_reason(understanding: TurnUnderstanding) -> str:
    return _HANDOFF_ESCALATIONS.get(understanding.intent, "policy_boundary")


def route_without_knowledge(  # noqa: C901
    understanding: TurnUnderstanding,
    profile: SiteProfile,
    *,
    visitor_words: Sequence[str],
    last_citations: Sequence[Citation],
    previous_was_clarification: bool,
    identifiers_found: bool,
    handoff_pending: bool = False,
) -> ResponseDecision | None:
    """The decision for a turn that needs no knowledge lookup, or None to run the pipeline."""
    reply = checked_reply(
        understanding.reply,
        visitor_words=tuple(visitor_words),
        off_brand_blocklist=profile.off_brand_blocklist,
        site_labels=(profile.name, profile.assistant_scope),
    )
    intent = understanding.intent
    contacts = contact_block(profile.contact_info)
    wants = wants_person(understanding, profile)

    if intent == "urgent_safety":
        # Emergency services first; a person is only offered (the visitor's next turn accepts).
        return ResponseDecision(
            ResponseOutcome.BOUNDARY,
            "urgent_safety",
            reply or URGENT_SAFETY_LINE,
            provider_status=ProviderStatus.OK,
            offer_handoff=profile.human_enabled and not handoff_pending,
        )
    if handoff_pending and intent == "handoff_decline":
        return ResponseDecision(
            ResponseOutcome.SYNTHESIZED_ANSWER,
            "handoff_declined",
            reply or KEEP_HELPING_LINE,
            provider_status=ProviderStatus.OK,
            cancel_handoff=True,
        )

    if intent == "shared_identifiers" and identifiers_found:
        return ResponseDecision(
            ResponseOutcome.BOUNDARY,
            "identifiers_shared",
            reply or IDENTIFIERS_LINE,
            provider_status=ProviderStatus.OK,
        )
    if intent in _BOUNDARY_REASONS:
        return ResponseDecision(
            ResponseOutcome.BOUNDARY,
            _BOUNDARY_REASONS[intent],
            reply or (OUT_OF_SCOPE if intent == "off_topic" else KEEP_HELPING_LINE),
            provider_status=ProviderStatus.OK,
        )
    if wants and handoff_pending:
        # A person is already on the way: acknowledge, never open a second handoff.
        return ResponseDecision(
            ResponseOutcome.SYNTHESIZED_ANSWER,
            "handoff_pending",
            reply or PENDING_LINE,
            provider_status=ProviderStatus.OK,
        )
    if wants:
        # With staff online the handoff joins the chat; without, it is a callback request the
        # team works from its queue (HandoffService picks the route from the site). Either way
        # the visitor also gets the configured contact route when nobody can join right now.
        body = "\n\n".join(
            part for part in (reply, "" if profile.human_enabled else contacts) if part
        )
        return ResponseDecision(
            ResponseOutcome.BOUNDARY,
            intent if intent in _HANDOFF_ESCALATIONS else "handoff_topic",
            body,
            provider_status=ProviderStatus.OK,
            handoff_reason=escalation_reason(understanding),
        )
    if intent in _CONVERSATIONAL_REASONS:
        # Conversational turns are never knowledge questions: when the model's text is unusable
        # a neutral line answers, and the turn is never counted as a knowledge gap.
        return ResponseDecision(
            ResponseOutcome.SYNTHESIZED_ANSWER,
            _CONVERSATIONAL_REASONS[intent],
            reply or KEEP_HELPING_LINE,
            provider_status=ProviderStatus.OK,
        )
    if intent == "contact_request" and contacts:
        return ResponseDecision(
            ResponseOutcome.SYNTHESIZED_ANSWER,
            "contact_route",
            "\n\n".join(part for part in (reply, contacts) if part),
            provider_status=ProviderStatus.OK,
        )
    if intent == "source_request":
        if last_citations:
            return source_followup_decision(list(last_citations), reply)
        if reply:
            return ResponseDecision(
                ResponseOutcome.CLARIFICATION,
                "source_followup",
                reply,
                provider_status=ProviderStatus.OK,
            )
        return None
    if (
        understanding.ambiguity.strip()
        and intent in KNOWLEDGE_INTENTS
        and reply
        and not previous_was_clarification
    ):
        return ResponseDecision(
            ResponseOutcome.CLARIFICATION,
            "clarify",
            reply,
            provider_status=ProviderStatus.OK,
        )
    return None
