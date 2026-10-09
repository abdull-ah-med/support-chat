"""Per-site assistant profile and free-text conversation intent."""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "57siteprofile"
down_revision = "56workspacesearch"
branch_labels = None
depends_on = None

LEGACY_INTENTS = "('pricing','turnaround','dot','fcra','portal','escalate','other')"


def upgrade() -> None:
    op.add_column(
        "sites", sa.Column("assistant_scope", sa.Text(), nullable=False, server_default="")
    )
    op.add_column(
        "sites",
        sa.Column(
            "handoff_topics",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "sites", sa.Column("primary_language", sa.Text(), nullable=False, server_default="en")
    )
    op.drop_constraint("ck_conversations_intent", "conversations", type_="check")
    op.create_check_constraint(
        "ck_conversations_intent_length",
        "conversations",
        "intent IS NULL OR char_length(intent) <= 64",
    )


def downgrade() -> None:
    op.drop_constraint("ck_conversations_intent_length", "conversations", type_="check")
    # Free-text values can no longer satisfy the legacy enum; fold them into 'other'.
    op.execute(
        f"UPDATE conversations SET intent = 'other' WHERE intent IS NOT NULL AND intent NOT IN {LEGACY_INTENTS}"
    )
    op.create_check_constraint(
        "ck_conversations_intent",
        "conversations",
        f"intent IS NULL OR intent IN {LEGACY_INTENTS}",
    )
    op.drop_column("sites", "primary_language")
    op.drop_column("sites", "handoff_topics")
    op.drop_column("sites", "assistant_scope")
