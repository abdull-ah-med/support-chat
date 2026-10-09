from dataclasses import dataclass
from typing import Literal

Kind = Literal["faq", "section", "table", "definition", "prose", "fact"]


@dataclass(frozen=True)
class EvidenceUnit:
    kind: Kind
    heading: str
    canonical_question: str | None
    answer_verbatim: str
    body_for_search: str
    display_locator: str | None
    aliases: tuple[str, ...] = ()
    topic: str | None = None
    structured_text: str | None = None
    enabled: bool = True
    review_note: str | None = None
    # Ancestor section titles ending with ``heading``; gives a chunk its place on the page.
    heading_path: tuple[str, ...] = ()
