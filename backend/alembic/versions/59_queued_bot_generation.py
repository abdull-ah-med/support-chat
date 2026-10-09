"""Let the assistant keep answering while a handoff is queued.

A queued conversation is waiting for a person to join; until one does, the assistant still
answers the visitor, so a generation may now be active in the ``queued`` state too. It stays
forbidden once a person owns the chat (``human``) or the chat is closed.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "59queuedbotgeneration"
down_revision: str | Sequence[str] | None = "58kbneutralsearch"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

NAME = "ck_conversations_generation_bot_only"


def upgrade() -> None:
    op.drop_constraint(NAME, "conversations", type_="check")
    op.create_check_constraint(
        NAME, "conversations", "(active_generation_id IS NULL) OR (state IN ('bot','queued'))"
    )


def downgrade() -> None:
    op.execute("UPDATE conversations SET active_generation_id = NULL WHERE state <> 'bot'")
    op.drop_constraint(NAME, "conversations", type_="check")
    op.create_check_constraint(
        NAME, "conversations", "(active_generation_id IS NULL) OR (state = 'bot')"
    )
