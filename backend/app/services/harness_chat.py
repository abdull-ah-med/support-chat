"""TEMPORARY: drive real visitor conversations through ConversationService for testers.

Same service path the widget uses (prechat, visitor_message, run_bot_turn) with real
providers. Remove with the rest of the harness.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.canned_reply import CannedReply
from app.models.conversation import Conversation
from app.models.handoff_context import HandoffContext
from app.models.kb_chunk import KbChunk
from app.models.kb_snapshot import KbSnapshot
from app.models.knowledge_gap import KnowledgeGapHit
from app.models.message import Message
from app.models.site import Site
from app.models.visitor import Visitor
from app.repositories.message_repo import MessageRepository
from app.repositories.site_repo import SiteRepository
from app.services.bot_trace import capture_trace, record_trace, wait_for_trace_tasks
from app.services.conversation_service import CommandError, ConversationService
from app.services.harness_store import get_store
from app.services.harness_telemetry import (
    capture_llm_calls,
    json_safe,
    summarize_trace,
)
from app.services.pii_redactor import redact_for_model

TOKEN_PREFIX = "harness-"
BACKGROUND_WAIT_SECONDS = 20.0


class HarnessError(Exception):
    def __init__(self, code: str, status_code: int = 409) -> None:
        self.code = code
        self.status_code = status_code
        super().__init__(code)


async def create_conversation(
    session: AsyncSession, site_key: str, name: str | None, email: str | None
) -> dict[str, Any]:
    site = await SiteRepository(session).get_by_key(site_key)
    if site is None or not site.enabled:
        raise HarnessError("not_found", 404)
    if not site.allowed_origins:
        raise HarnessError("site_has_no_allowed_origin")
    visitor = Visitor(site_id=site.id, resume_token_hash=TOKEN_PREFIX + uuid4().hex)
    session.add(visitor)
    await session.flush()
    conversation = Conversation(site_id=site.id, visitor_id=visitor.id, state="prechat")
    session.add(conversation)
    await session.commit()
    try:
        await ConversationService(session).submit_prechat(
            conversation.id,
            visitor.id,
            site.allowed_origins[0],
            uuid4(),
            name or "Harness Tester",
            email or "harness-tester@supportchat.local",
            "",
            "other",
            "",
        )
    except CommandError as exc:
        await session.rollback()
        raise HarnessError(exc.code) from None
    await session.refresh(conversation)
    return {
        "conversation_id": str(conversation.id),
        "site_key": site.key,
        "state": conversation.state,
        "bot_enabled": site.bot_enabled,
    }


def _message_row(message: Message) -> dict[str, Any]:
    return {
        "id": message.id,
        "role": message.role,
        "body": message.body,
        "system_reason": message.system_reason,
        "outcome": message.response_outcome,
        "reason_code": message.response_reason_code,
        "created_at": message.created_at.isoformat() if message.created_at else None,
    }


async def _citations(
    session: AsyncSession, site_id: UUID, rows: list[Message]
) -> list[dict[str, Any]]:
    chunk_ids = {c.chunk_id for m in rows for c in m.citations if c.chunk_id}
    chunk_site: dict[UUID, UUID] = {}
    if chunk_ids:
        found = await session.execute(
            select(KbChunk.id, KbChunk.site_id).where(KbChunk.id.in_(chunk_ids))
        )
        chunk_site = {cid: sid for cid, sid in found.all()}
    out = []
    for message in rows:
        for c in message.citations:
            owner = chunk_site.get(c.chunk_id) if c.chunk_id else None
            out.append(
                {
                    "message_id": message.id,
                    "chunk_id": str(c.chunk_id) if c.chunk_id else None,
                    "source_url": c.source_url,
                    "source_title": c.source_title,
                    "cited_text": (c.cited_text or "")[:400],
                    "citation_site_matches": c.site_id == site_id,
                    "chunk_site_matches": owner == site_id if owner else None,
                    # A citation with a missing chunk cannot be proven in-site: count as foreign.
                    "belongs_to_site": c.site_id == site_id and owner == site_id,
                }
            )
    return out


async def _count(session: AsyncSession, model: Any, conversation_id: UUID) -> int:
    result = await session.execute(
        select(func.count()).select_from(model).where(model.conversation_id == conversation_id)
    )
    return int(result.scalar_one())


async def send_message(session: AsyncSession, conversation_id: UUID, text: str) -> dict[str, Any]:
    conversation = await session.get(Conversation, conversation_id)
    visitor = await session.get(Visitor, conversation.visitor_id) if conversation else None
    if (
        conversation is None
        or visitor is None
        or not visitor.resume_token_hash.startswith(TOKEN_PREFIX)
    ):
        raise HarnessError("not_found", 404)
    site = await session.get(Site, conversation.site_id)
    assert site is not None
    # Read once: the service commits during the turn, which expires the loaded site row.
    site_id, site_key = site.id, site.key
    messages = MessageRepository(session)
    latest = await messages.latest(conversation.id)
    cursor = latest.id if latest is not None else 0
    state_before = conversation.state
    handoffs_before = await _count(session, HandoffContext, conversation.id)
    gaps_before = await _count(session, KnowledgeGapHit, conversation.id)
    started = time.perf_counter_ns()
    error: str | None = None
    generation_id = None

    with capture_trace() as trace, capture_llm_calls() as llm_calls:
        record_trace("request", message=redact_for_model(text), site_key=site_key)
        service = ConversationService(session)
        try:
            result = await service.visitor_message(
                conversation.id, visitor.id, site.allowed_origins[0], uuid4(), text
            )
            generation_id = result.generation_id
            if generation_id is not None:
                await service.run_bot_turn(conversation.id, generation_id)
        except CommandError as exc:
            await session.rollback()
            raise HarnessError(exc.code, 429 if exc.code == "rate_limited" else 409) from None
        except Exception as exc:  # report pipeline failures as telemetry, not a 500
            await session.rollback()
            error = f"{type(exc).__name__}: {redact_for_model(str(exc))[:300]}"
        pending = await wait_for_trace_tasks(timeout=BACKGROUND_WAIT_SECONDS)
        record_trace("background_work", pending=pending)
        latency_ms = (time.perf_counter_ns() - started) // 1_000_000

    await session.refresh(conversation)
    rows = await messages.list_after(conversation.id, cursor)
    replies = [r for r in rows if r.role in {"bot", "system"}]
    safe_trace = json_safe(trace)
    telemetry = summarize_trace(safe_trace, llm_calls)
    if error:
        telemetry["errors"].append({"section": "harness", "error": error})
    citations = await _citations(session, site_id, replies)
    handoffs_after = await _count(session, HandoffContext, conversation.id)
    gaps_after = await _count(session, KnowledgeGapHit, conversation.id)
    last = replies[-1] if replies else None
    turn = {
        "conversation_id": str(conversation.id),
        "site_key": site_key,
        "turn_at": datetime.now(UTC).isoformat(),
        "visitor_text": redact_for_model(text),
        "replies": [_message_row(r) for r in replies],
        "reply_text": "\n\n".join(r.body for r in replies),
        "state_before": state_before,
        "state": conversation.state,
        "intent": conversation.intent,
        "fallback_count": conversation.fallback_count,
        "outcome": last.response_outcome if last else None,
        "reason_code": (last.response_reason_code or last.system_reason) if last else None,
        "citations": citations,
        "handoff_opened": handoffs_after > handoffs_before,
        "knowledge_gap_recorded": gaps_after > gaps_before,
        "generation_id": str(generation_id) if generation_id else None,
        "latency_ms": latency_ms,
        "error": error,
        "telemetry": telemetry,
        "trace": safe_trace,
    }
    get_store().add(turn)
    return turn


async def transcript(session: AsyncSession, conversation_id: UUID) -> dict[str, Any]:
    conversation = await session.get(Conversation, conversation_id)
    visitor = await session.get(Visitor, conversation.visitor_id) if conversation else None
    if (
        conversation is None
        or visitor is None
        or not visitor.resume_token_hash.startswith(TOKEN_PREFIX)
    ):
        raise HarnessError("not_found", 404)
    rows = await MessageRepository(session).list_after(conversation.id, 0)
    return {
        "conversation_id": str(conversation.id),
        "state": conversation.state,
        "intent": conversation.intent,
        "messages": [_message_row(r) for r in rows],
        "turns": get_store().for_conversation(str(conversation.id)),
    }


async def list_sites(session: AsyncSession) -> list[dict[str, Any]]:
    out = []
    for site in await SiteRepository(session).list_all():
        live = select(KbSnapshot.id).where(
            KbSnapshot.site_id == site.id, KbSnapshot.state == "live"
        )
        chunks = await session.execute(
            select(func.count())
            .select_from(KbChunk)
            .where(
                KbChunk.site_id == site.id, KbChunk.enabled.is_(True), KbChunk.snapshot_id.in_(live)
            )
        )
        headings = await session.execute(
            select(KbChunk.heading)
            .where(
                KbChunk.site_id == site.id, KbChunk.enabled.is_(True), KbChunk.snapshot_id.in_(live)
            )
            .order_by(KbChunk.ordinal)
            .limit(200)
        )
        unique = list(dict.fromkeys(h for (h,) in headings.all() if h))
        step = max(1, len(unique) // 30)
        canned = await session.execute(
            select(func.count())
            .select_from(CannedReply)
            .where(
                CannedReply.site_id == site.id,
                CannedReply.enabled.is_(True),
                CannedReply.bot_eligible.is_(True),
            )
        )
        out.append(
            {
                "key": site.key,
                "name": site.name,
                "enabled": site.enabled,
                "bot_enabled": site.bot_enabled,
                "human_enabled": site.human_enabled,
                "assistant_scope": site.assistant_scope,
                "primary_language": site.primary_language,
                "live_chunks": int(chunks.scalar_one()),
                "canned_bot_eligible": int(canned.scalar_one()),
                "sample_headings": unique[::step][:30],
            }
        )
    return out
