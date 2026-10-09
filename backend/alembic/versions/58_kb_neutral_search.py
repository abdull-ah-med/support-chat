"""Language-neutral full-text search for knowledge chunks and canned replies.

The generated ``search_document`` columns used the ``english`` configuration, which
stems and drops stop words for one language only. They now use ``simple`` over text
folded by ``workspace_search_fold`` (migration 56: NFKD, accents removed, lowered), so
Spanish, French or any other language tokenises the same way on both sides of a query.

Canned replies stop indexing their shortcut slug: a reply is found by the phrases it
answers (aliases) and its body. Their stored vectors were built from the old text, so
they are cleared and the worker re-embeds them.
"""

from collections.abc import Sequence

from alembic import op

revision: str = "58kbneutralsearch"
down_revision: str | Sequence[str] | None = "57siteprofile"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

CHUNK_NEUTRAL = (
    "setweight(to_tsvector('simple', workspace_search_fold(coalesce(canonical_question, ''))), 'A') || "
    "setweight(to_tsvector('simple', workspace_search_fold(kb_aliases_as_text(aliases))), 'A') || "
    "setweight(to_tsvector('simple', workspace_search_fold(coalesce(heading, ''))), 'B') || "
    "setweight(to_tsvector('simple', workspace_search_fold(coalesce(body, ''))), 'C')"
)
CHUNK_ENGLISH = (
    "setweight(to_tsvector('english', coalesce(canonical_question, '')), 'A') || "
    "setweight(to_tsvector('english', kb_aliases_as_text(aliases)), 'A') || "
    "setweight(to_tsvector('english', coalesce(heading, '')), 'B') || "
    "setweight(to_tsvector('english', coalesce(body, '')), 'C')"
)
CANNED_NEUTRAL = (
    "setweight(to_tsvector('simple', workspace_search_fold(canned_aliases_as_text(aliases))), 'A') || "
    "setweight(to_tsvector('simple', workspace_search_fold(coalesce(body, ''))), 'B')"
)
CANNED_ENGLISH = (
    "setweight(to_tsvector('english', coalesce(shortcut, '')), 'A') || "
    "setweight(to_tsvector('english', canned_aliases_as_text(aliases)), 'A') || "
    "setweight(to_tsvector('english', coalesce(body, '')), 'B')"
)
TRGM_PLAIN = (
    "lower(coalesce(canonical_question, '') || ' ' || kb_aliases_as_text(aliases) || ' ' || "
    "coalesce(heading, '') || ' ' || coalesce(topic_label, ''))"
)
TRGM_FOLDED = (
    "workspace_search_fold(coalesce(canonical_question, '') || ' ' || "
    "kb_aliases_as_text(aliases) || ' ' || coalesce(heading, '') || ' ' || "
    "coalesce(topic_label, ''))"
)


def _swap_search_document(table: str, index: str, expression: str) -> None:
    # Dropping the column drops its GIN index with it.
    op.execute(f"ALTER TABLE {table} DROP COLUMN search_document")
    op.execute(
        f"ALTER TABLE {table} ADD COLUMN search_document tsvector "
        f"GENERATED ALWAYS AS ({expression}) STORED"
    )
    with op.get_context().autocommit_block():
        op.execute(f"CREATE INDEX CONCURRENTLY {index} ON {table} USING gin (search_document)")


def _swap_trigram_index(expression: str) -> None:
    with op.get_context().autocommit_block():
        op.execute("DROP INDEX CONCURRENTLY IF EXISTS ix_kb_chunks_grounded_trgm")
        op.execute(
            "CREATE INDEX CONCURRENTLY ix_kb_chunks_grounded_trgm "
            f"ON kb_chunks USING gin (({expression}) gin_trgm_ops)"
        )


def upgrade() -> None:
    _swap_search_document("kb_chunks", "ix_kb_chunks_search", CHUNK_NEUTRAL)
    _swap_search_document("canned_replies", "ix_canned_replies_search", CANNED_NEUTRAL)
    _swap_trigram_index(TRGM_FOLDED)
    op.execute("UPDATE canned_replies SET embedding = NULL, embedder_id = NULL")


def downgrade() -> None:
    _swap_trigram_index(TRGM_PLAIN)
    _swap_search_document("canned_replies", "ix_canned_replies_search", CANNED_ENGLISH)
    _swap_search_document("kb_chunks", "ix_kb_chunks_search", CHUNK_ENGLISH)
    op.execute("UPDATE canned_replies SET embedding = NULL, embedder_id = NULL")
