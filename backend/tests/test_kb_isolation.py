import uuid

from sqlalchemy import select

from app.db import session_maker
from app.llm.bot_responder import BotResponder
from app.llm.prompts import document_body
from app.models.message import Message
from app.repositories.site_repo import SiteRepository
from app.services.conversation_service import ConversationService
from app.services.kb_search import KbSearch
from tests.bot_fixtures import (
    EASY_BODY,
    EASY_TITLE,
    FALLBACK,
    FAST_QUERY,
    FCRA_BODY,
    FCRA_TITLE,
    INJECTION_QUERY,
    SCRIPTED_ANSWER,
    RecordingResponder,
    insert_bot_conversation,
    seed_brand_articles,
)
from tests.fake_anthropic import Draft, FakeAnthropic
from tests.fakes import FakeEmbedder
from tests.ws_helpers import HOST_ORIGIN, message_count


async def test_samplesite_paraphrase_selects_timing_article_and_prompt_excludes_fcra(
    migrated_db,
) -> None:
    async with session_maker()() as session:
        easy, _bg, timing, _fcra = await seed_brand_articles(session)
        hits = await KbSearch(session).search(easy.id, FAST_QUERY)
        bodies = "\n".join(document_body(hit) for hit in hits)
        titles = [getattr(hit, "title", None) or getattr(hit, "heading", None) for hit in hits]
        assert titles == [EASY_TITLE]
        assert [hit.id for hit in hits] == [timing.chunk_id]
        assert EASY_BODY in bodies
        assert FCRA_BODY not in bodies
        assert FCRA_TITLE not in bodies


async def test_verification_checks_same_query_never_receives_samplesite_source(
    migrated_db,
) -> None:
    responder = RecordingResponder()
    async with session_maker()() as session:
        _easy, bg, _timing, _fcra = await seed_brand_articles(session)
        visitor, conversation = await insert_bot_conversation(session, bg)
        await session.commit()
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            FAST_QUERY,
        )
        if result.generation_id is not None:
            await service.run_bot_turn(conversation.id, result.generation_id)
        conversation_id = conversation.id
        bg_id = bg.id

    async with session_maker()() as session:
        hits = await KbSearch(session).search(bg_id, FAST_QUERY)
    assert hits == []
    assert message_count(conversation_id, role="bot") == 1
    assert message_count(conversation_id, role="system", body=FALLBACK) == 0
    assert message_count(conversation_id, role="system", body=SCRIPTED_ANSWER) == 0
    assert message_count(conversation_id, body=EASY_BODY) == 0
    assert responder.draft_calls() == []


async def test_scripted_grounded_answer_writes_one_bot_row_with_samplesite_article_id(
    migrated_db,
) -> None:
    responder = RecordingResponder(answer=SCRIPTED_ANSWER)
    async with session_maker()() as session:
        easy, _bg, timing, _fcra = await seed_brand_articles(session)
        visitor, conversation = await insert_bot_conversation(session, easy)
        await session.commit()
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        result = await service.visitor_message(
            conversation.id,
            visitor.id,
            HOST_ORIGIN,
            uuid.uuid4(),
            FAST_QUERY,
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation.id, result.generation_id)
        conversation_id = conversation.id
        timing_id = timing.chunk_id

    assert len(responder.draft_calls()) == 1
    assert responder.draft_calls()[0]["article_ids"] == [timing_id]
    assert message_count(conversation_id, role="bot", body=SCRIPTED_ANSWER) == 1
    assert message_count(conversation_id, role="bot") == 1
    async with session_maker()() as session:
        row = (
            await session.execute(
                select(Message).where(
                    Message.conversation_id == conversation_id, Message.role == "bot"
                )
            )
        ).scalar_one()
        assert list(row.source_chunk_ids) == [timing_id]


async def test_injection_cannot_add_foreign_articles(migrated_db) -> None:
    # The model reads the message as an injection attempt: a boundary reply, no draft.
    responder = RecordingResponder(
        answer=SCRIPTED_ANSWER,
        understand={"intent": "prompt_injection", "reply": "I can only help with this site."},
    )
    async with session_maker()() as session:
        easy, _bg, timing, _fcra = await seed_brand_articles(session)
        visitor, conversation = await insert_bot_conversation(session, easy)
        await session.commit()
        conversation_id = conversation.id
        visitor_id = visitor.id
        timing_id = timing.chunk_id
        service = ConversationService(session, responder=responder, embedder=FakeEmbedder())
        result = await service.visitor_message(
            conversation_id,
            visitor_id,
            HOST_ORIGIN,
            uuid.uuid4(),
            INJECTION_QUERY,
        )
        if result.generation_id is not None:
            await service.run_bot_turn(conversation_id, result.generation_id)

    async with session_maker()() as session:
        easy_site = await SiteRepository(session).get_by_key("samplesite")
        assert easy_site is not None
        hits = await KbSearch(session).search(easy_site.id, INJECTION_QUERY)
        assert [article.title for article in hits] != [FCRA_TITLE]
        assert all(article.title != FCRA_TITLE for article in hits)
        bots = (
            (
                await session.execute(
                    select(Message).where(
                        Message.conversation_id == conversation_id, Message.role == "bot"
                    )
                )
            )
            .scalars()
            .all()
        )
        for row in bots:
            sources = list(row.source_chunk_ids or [])
            assert FCRA_TITLE not in (row.source_title or "")
            assert all(source_id == timing_id for source_id in sources)
            # Injection turns must never cite a foreign-site chunk.
            assert timing_id not in sources or row.response_outcome in {
                None,
                "exact_answer",
                "synthesized_answer",
                "partial_answer",
            }
        assert bots
        assert all(
            row.response_outcome == "boundary" or (row.source_chunk_ids or []) == [timing_id]
            for row in bots
        )
    assert responder.draft_calls() == []


async def test_default_responder_persists_recorded_anthropic_answer(migrated_db) -> None:
    client = FakeAnthropic(draft=Draft(SCRIPTED_ANSWER, EASY_BODY))
    async with session_maker()() as session:
        easy, _bg, timing, _fcra = await seed_brand_articles(session)
        visitor, conversation = await insert_bot_conversation(session, easy)
        await session.commit()
        conversation_id = conversation.id
        visitor_id = visitor.id
        timing_id = timing.chunk_id
        service = ConversationService(
            session, responder=BotResponder(client=client), embedder=FakeEmbedder()
        )
        result = await service.visitor_message(
            conversation_id,
            visitor_id,
            HOST_ORIGIN,
            uuid.uuid4(),
            FAST_QUERY,
        )
        assert result.generation_id is not None
        await service.run_bot_turn(conversation_id, result.generation_id)

    # Only the requesting site's evidence reaches the provider.
    documents = client.stage_calls("draft")[0]["messages"]
    assert EASY_BODY in str(documents)
    assert FCRA_BODY not in str(documents)
    assert message_count(conversation_id, role="bot", body=SCRIPTED_ANSWER) == 1
    assert message_count(conversation_id, role="bot") == 1
    async with session_maker()() as session:
        row = (
            await session.execute(
                select(Message).where(
                    Message.conversation_id == conversation_id, Message.role == "bot"
                )
            )
        ).scalar_one()
        assert list(row.source_chunk_ids) == [timing_id]
