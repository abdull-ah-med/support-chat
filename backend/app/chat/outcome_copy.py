"""Fixed system copy for handoff and failure states.

Deliberately small, brand-neutral and domain-neutral. Conversational replies (greetings,
redirects, boundaries, "I don't know") are written by the model in the visitor's language;
only text that no model call can produce lives here.
"""

from __future__ import annotations

WAITING_LINE = (
    "A specialist has been asked to join this chat. You can keep asking questions while you wait."
)
CALLBACK_LINE = "A representative will contact you in 24 hours."
KEEP_HELPING_LINE = "I'm still here to help. What would you like to know?"

# Last resort when the model's own "I don't know" text is unusable. Still honest, still neutral.
INSUFFICIENT_HUMAN = (
    "I don't have verified information on that. Would you like me to ask a specialist to join?"
)
INSUFFICIENT_SOLO = "I don't have verified information on that."
TECH_FAIL_HUMAN = (
    "Sorry, I hit a temporary issue. Please try again, or I can ask a specialist to join."
)
TECH_FAIL_SOLO = "Sorry, I hit a temporary issue. Please try again in a moment."
POLICY_BOUNDARY = "I can't share that in chat. A specialist has been asked to walk you through it."
OUT_OF_SCOPE = "I can't help with that here."

SENSITIVE_HUMAN = "A specialist has been asked to help through a secure channel."
INDIVIDUAL_CASE_HUMAN = (
    "A specialist is required to discuss an individual's case and has been asked to join this chat."
)
RETRIEVAL_MISS_HUMAN = INSUFFICIENT_HUMAN
SUFFICIENCY_FAIL_HUMAN = (
    "I don't have enough information here to answer that safely. A specialist can help."
)
PROVIDER_TIMEOUT_HUMAN = TECH_FAIL_HUMAN
REPEATED_MISS_HUMAN = "I've asked a specialist to join so you don't have to repeat yourself."
RATE_CEILING_HUMAN = "I've hit a temporary limit. I've asked a specialist to join."


def callback_line(window_hours: int) -> str:
    hours = max(1, int(window_hours))
    if hours == 24:
        return CALLBACK_LINE
    unit = "hour" if hours == 1 else "hours"
    return f"A representative will contact you in {hours} {unit}."


def contact_block(contact_info: list[str] | tuple[str, ...] | None) -> str:
    """The configured contact route, verbatim, one entry per line (empty when none is set)."""
    return "\n".join(item.strip() for item in (contact_info or ()) if item and item.strip())


def handoff_copy(  # noqa: C901
    reason: str,
    *,
    human_enabled: bool,
    window_hours: int = 24,
) -> str:
    callback = callback_line(window_hours)
    if reason == "visitor_request":
        return WAITING_LINE if human_enabled else callback
    if reason == "sensitive":
        return SENSITIVE_HUMAN if human_enabled else callback
    if reason == "individual_case":
        if human_enabled:
            return INDIVIDUAL_CASE_HUMAN
        return "A specialist is required to discuss an individual's case. " + callback
    if reason == "retrieval_miss":
        if human_enabled:
            return RETRIEVAL_MISS_HUMAN
        return "I don't have verified information on that. " + callback
    if reason == "sufficiency_fail":
        if human_enabled:
            return SUFFICIENCY_FAIL_HUMAN
        return "I don't have enough information here to answer that safely. " + callback
    if reason == "provider_timeout":
        if human_enabled:
            return PROVIDER_TIMEOUT_HUMAN
        return "Sorry, I hit a temporary issue. " + callback
    if reason == "repeated_miss":
        if human_enabled:
            return REPEATED_MISS_HUMAN
        return "I'm passing this on so you don't have to repeat yourself. " + callback
    if reason == "policy_boundary":
        if human_enabled:
            return POLICY_BOUNDARY
        return "I can't share that in chat. " + callback
    if reason == "rate_ceiling":
        if human_enabled:
            return RATE_CEILING_HUMAN
        return "I've hit a temporary limit. " + callback
    if reason == "off_topic":
        return OUT_OF_SCOPE
    raise ValueError(f"unsupported handoff copy: {reason}")
