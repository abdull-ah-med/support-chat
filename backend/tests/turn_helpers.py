"""Drive whole bot turns through ConversationService with a fake Anthropic client."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.llm.bot_responder import BotResponder
from app.models.conversation import Conversation
from app.models.knowledge_gap import KnowledgeGapHit
from app.models.message import Message
from app.models.site import Site
from app.models.visitor import Visitor
from app.services.conversation_service import ConversationService
from tests.bot_fixtures import insert_bot_conversation, insert_chunk, insert_site
from tests.fakes import FakeEmbedder
from tests.ws_helpers import HOST_ORIGIN


@dataclass
class Chat:
    session: AsyncSession
    site: Site
    visitor: Visitor
    conversation: Conversation
    service: ConversationService

    async def say(self, text: str) -> None:
        result = await self.service.visitor_message(
            self.conversation.id, self.visitor.id, HOST_ORIGIN, uuid.uuid4(), text
        )
        if result.generation_id is not None:
            await self.service.run_bot_turn(self.conversation.id, result.generation_id)
        await self.session.refresh(self.conversation)

    async def rows(self, role: str | None = None) -> list[Message]:
        query = select(Message).where(Message.conversation_id == self.conversation.id)
        if role is not None:
            query = query.where(Message.role == role)
        return list((await self.session.scalars(query.order_by(Message.id))).all())

    async def last_bot(self) -> Message:
        rows = await self.rows("bot")
        assert rows, "the assistant did not reply"
        return rows[-1]

    async def gap_hits(self) -> list[KnowledgeGapHit]:
        return list(
            (
                await self.session.scalars(
                    select(KnowledgeGapHit).where(KnowledgeGapHit.site_id == self.site.id)
                )
            ).all()
        )


async def open_chat(
    session: AsyncSession,
    client,
    *,
    key: str = "flow-site",
    name: str = "Flow Site",
    chunks: list[tuple[str, str]] | None = None,
    **site_fields,
) -> Chat:
    """A bot-state chat on a fresh site. `chunks` are (title, body) pairs of live knowledge."""
    site = await insert_site(session, key, name)
    for field_name, value in site_fields.items():
        setattr(site, field_name, value)
    for title, body in chunks or []:
        await insert_chunk(session, site, title, body)
    visitor, conversation = await insert_bot_conversation(session, site)
    await session.commit()
    responder = client if hasattr(client, "understand_turn") else BotResponder(client=client)
    service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
    return Chat(session, site, visitor, conversation, service)
