"""Native SDK citations become offsets the validator can check; unsafe text never passes."""

import uuid
from types import SimpleNamespace

from app.llm.bot_responder import extract_native_citations
from app.services import output_validator

EASY_BODY = "Most negative results are reported within 24-48 hours."
TIMING_ID = uuid.UUID("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa")
TIMING_URL = "https://example.test/turnaround"


def _hit(chunk_id: uuid.UUID = TIMING_ID):
    return SimpleNamespace(
        id=chunk_id, title="Turnaround", heading="", body=EASY_BODY, url=TIMING_URL
    )


def _block(text: str, **citation):
    return SimpleNamespace(type="text", text=text, citations=[SimpleNamespace(**citation)])


def test_native_citation_survives_response_whitespace_trimming() -> None:
    body, citations = extract_native_citations(
        [
            _block(
                f"\n{EASY_BODY}\n",
                document_index=0,
                start_char_index=0,
                end_char_index=len(EASY_BODY),
                cited_text=EASY_BODY,
            )
        ],
        [_hit()],
    )
    assert body == EASY_BODY
    assert len(citations) == 1
    assert (citations[0].response_start, citations[0].response_end) == (0, len(body))
    assert citations[0].chunk_id == TIMING_ID
    assert citations[0].source_url == TIMING_URL


def test_citation_whose_text_is_not_the_source_is_dropped() -> None:
    _body, citations = extract_native_citations(
        [
            _block(
                EASY_BODY,
                document_index=0,
                start_char_index=0,
                end_char_index=10,
                cited_text="forged text",
            )
        ],
        [_hit()],
    )
    assert citations == []


def test_citation_to_a_document_that_was_not_sent_is_dropped() -> None:
    _body, citations = extract_native_citations(
        [
            _block(
                EASY_BODY,
                document_index=3,
                start_char_index=0,
                end_char_index=len(EASY_BODY),
                cited_text=EASY_BODY,
            )
        ],
        [_hit()],
    )
    assert citations == []


def test_output_validator_rejects_script_and_accepts_plain_text() -> None:
    def check(text: str):
        return output_validator.evaluate(text, allowed_ids=[TIMING_ID], cited=[TIMING_ID])

    assert check("<script>alert(1)</script>").accepted is False
    assert check(EASY_BODY).accepted is True


def test_output_validator_accepts_a_less_than_comparison_and_a_link_mention() -> None:
    text = "Results take < 48 hours; see https://example.test/turnaround for details."
    assert output_validator.evaluate(text, allowed_ids=[TIMING_ID], cited=[TIMING_ID]).accepted
