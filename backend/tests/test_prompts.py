import re

from app.llm.answer_relevance import ASSESS_RULES
from app.llm.prompts import (
    NO_ANSWER,
    REPAIR_RULES,
    SYSTEM_RULES,
    system_rules_for,
    visitor_turn_text,
)
from app.llm.understanding import UNDERSTAND_RULES
from app.services.full_context import STAFF_MARKER

# Prompts are site-neutral: what a site is about arrives as data (name, scope, handoff topics),
# never as an example or a phrase list inside the rules.
SITE_SPECIFIC = re.compile(
    r"\b(?:employer|candidate|drug|alcohol|DOT|FCRA|HIPAA|screening|collection site|"
    r"verification check|SampleSite|247)\b",
    re.IGNORECASE,
)
ALL_RULES = {
    "draft": SYSTEM_RULES,
    "repair": REPAIR_RULES,
    "understand": UNDERSTAND_RULES,
    "assess": ASSESS_RULES,
}


def test_system_rules_include_site_name_scope_and_contract_blocks() -> None:
    rules = system_rules_for("Test Brand", "lab testing questions")
    assert "Test Brand" in rules
    assert "lab testing questions" in rules
    for tag in (
        "<role>",
        "<language>",
        "<grounding>",
        "<conversation>",
        "<response_style>",
        "<safety>",
    ):
        assert tag in rules


def test_scope_is_optional_and_bounded() -> None:
    assert "covers" not in system_rules_for("Acme")
    assert len(system_rules_for("Acme", "x" * 5000)) < len(system_rules_for("Acme")) + 700


def test_rules_name_no_domain_brand_or_language_specific_phrases() -> None:
    for name, text in ALL_RULES.items():
        assert not SITE_SPECIFIC.search(text), (name, SITE_SPECIFIC.search(text).group())
        # No fixed opener lists in any language: limitations are judged by meaning.
        assert "I cannot confirm" not in text
        assert "No puedo" not in text


def test_drafting_rules_define_the_no_answer_sentinel_and_mirror_the_language() -> None:
    rules = system_rules_for("Acme")
    assert f"exactly {NO_ANSWER}" in rules
    assert "language of the visitor's latest message" in rules
    assert NO_ANSWER in REPAIR_RULES


def test_every_stage_knows_staff_turns_are_not_the_bot() -> None:
    for name in ("draft", "understand", "assess"):
        assert STAFF_MARKER.strip() in ALL_RULES[name], name


def test_untrusted_data_rule_is_stated_for_each_model_call() -> None:
    assert "untrusted" in system_rules_for("Acme")
    assert "untrusted" in UNDERSTAND_RULES
    assert "untrusted" in ASSESS_RULES


def test_unnamed_sites_are_called_this_brand_and_named_ones_are_prefixed() -> None:
    assert "this brand" in system_rules_for("")
    assert visitor_turn_text("", "What is SampleMail?") == "What is SampleMail?"
    assert "Brand: Acme" in visitor_turn_text("Acme", "How fast are results?")
