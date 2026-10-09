"""Retrieval is language-neutral, site-scoped in every arm, and says when nothing is close."""

import pytest
from sqlalchemy import select

from app.db import session_maker
from app.models.kb_chunk import KbChunk
from app.services.kb_hybrid import COSINE_FLOOR, HybridKbSearch
from tests.bot_fixtures import insert_site
from tests.fakes import MappedEmbedder, UnavailableEmbedder, unit_vector
from tests.kb_corpus import ChunkSpec, make_corpus

SHIPPING = ChunkSpec(
    answer="El envío tarda dos días hábiles en llegar a cualquier ciudad.",
    heading="Plazos de envío",
    question="¿Cuánto tarda el envío?",
)
RETURNS = ChunkSpec(
    answer="Aceptamos devoluciones durante treinta días con el recibo.",
    heading="Devoluciones",
    question="¿Cómo devuelvo un producto?",
)


async def _hits(session, site, text: str, vector=None, **kwargs):
    return await HybridKbSearch(session).search(site.id, text, vector, **kwargs)


async def test_spanish_question_without_accents_finds_accented_text(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, [[SHIPPING, RETURNS]])
        await session.commit()
        shipping = corpus.chunks[0][0]
        for query in (
            "tiempo de envio dos dias habiles",  # lexical arm, no exact match
            "PLAZOS DE ENVÍO",
            "cuanto tarda el envio",  # exact match on the stored question
        ):
            hits = await _hits(session, site, query)
            assert hits and hits[0].id == shipping.id, query


async def test_accented_question_finds_unaccented_text(migrated_db) -> None:
    plain = ChunkSpec(
        answer="El envio tarda dos dias habiles en llegar a cualquier ciudad.",
        heading="Plazos de envio",
    )
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, [[plain, RETURNS]])
        await session.commit()
        hits = await _hits(session, site, "¿Cuántos días hábiles tarda el envío?")
    assert hits and hits[0].id == corpus.chunks[0][0].id


async def test_word_families_match_without_a_language_specific_stemmer(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(
            session,
            site,
            [[ChunkSpec(answer="Los resultados llegan por correo.", heading="Entrega")]],
        )
        await session.commit()
        for query in ("resultado correo", "resultados correo"):
            hits = await _hits(session, site, query)
            assert [hit.id for hit in hits] == [corpus.chunks[0][0].id], query


async def test_a_phrase_with_every_term_outranks_one_sharing_a_common_word(migrated_db) -> None:
    both = ChunkSpec(answer="Pago con tarjeta de crédito o débito.", heading="Medios de pago")
    one = ChunkSpec(answer="Tarjeta de regalo para amigos.", heading="Regalos")
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, [[one, both]])
        await session.commit()
        hits = await _hits(session, site, "pago con tarjeta de credito")
    assert hits[0].id == corpus.chunks[0][1].id


async def test_every_retrieval_arm_stays_inside_the_requesting_site(migrated_db) -> None:
    vector = unit_vector(7)
    twin = ChunkSpec(
        answer="El envío tarda dos días hábiles.",
        heading="Plazos de envío",
        question="¿Cuánto tarda el envío?",
        vector=vector,
    )
    crowd = [
        ChunkSpec(
            answer=f"Plazo de envío número {index} para pedidos grandes.",
            heading=f"Plazos de envío {index}",
            vector=unit_vector(7),
        )
        for index in range(30)
    ]
    async with session_maker()() as session:
        alpha = await insert_site(session, "alpha", "Alpha")
        beta = await insert_site(session, "beta", "Beta")
        alpha_corpus = await make_corpus(session, alpha, [[twin]])
        beta_corpus = await make_corpus(session, beta, [[twin, *crowd]])
        await session.commit()
        alpha_ids = {chunk.id for rows in alpha_corpus.chunks for chunk in rows}
        beta_ids = {chunk.id for rows in beta_corpus.chunks for chunk in rows}
        search = HybridKbSearch(session)
        cases = {
            "exact": ("¿Cuánto tarda el envío?", None),
            "lexical": ("plazos envio dos dias", None),
            "trigram": ("plazs de enviio", None),
            "dense": ("zzz qqq", vector),
        }
        for arm, (query, query_vector) in cases.items():
            hits = await search.search(alpha.id, query, query_vector)
            assert {hit.id for hit in hits} == alpha_ids, arm
            assert not beta_ids & {hit.id for hit in hits}, arm
            assert all(hit.site_id == alpha.id for hit in hits), arm
        # The other site's closer vectors must not starve this site's nearest-neighbour scan.
        beta_hits = await search.search(beta.id, "zzz qqq", vector)
        assert beta_hits and not alpha_ids & {hit.id for hit in beta_hits}


async def test_exact_question_match_leads_and_keeps_supporting_context(migrated_db) -> None:
    class CountingEmbedder(MappedEmbedder):
        calls = 0

        async def embed_query(self, text: str):
            type(self).calls += 1
            return await super().embed_query(text)

    exact = ChunkSpec(
        answer="Restablece la clave desde Ajustes, opción Seguridad.",
        heading="Clave",
        question="¿Cómo restablezco mi clave?",
        vector=unit_vector(5),
    )
    related = ChunkSpec(
        answer="La clave debe tener doce caracteres y una cifra; restablecer la clave exige correo.",
        heading="Requisitos de la clave",
        vector=unit_vector(5),
    )
    unrelated = ChunkSpec(
        answer="Nuestro horario es de lunes a viernes; la clave no cambia.",
        heading="Horario",
        vector=unit_vector(9),
    )
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, [[unrelated, related, exact]])
        await session.commit()
        embedder = CountingEmbedder({})
        hits = await HybridKbSearch(session).search_with_deferred_embed(
            embedder, site.id, "como restablezco mi clave"
        )
    assert hits[0].id == corpus.chunks[0][2].id
    assert corpus.chunks[0][1].id in {hit.id for hit in hits}, "supporting passages are kept"
    assert corpus.chunks[0][0].id not in {hit.id for hit in hits}
    assert hits.best_cosine == 1.0
    assert CountingEmbedder.calls == 0, "an exact question match needs no embedding"


async def test_best_cosine_tells_the_caller_when_nothing_is_close(migrated_db) -> None:
    embedder = MappedEmbedder({"envío": 1, "envio": 1, "pan": 2})
    on_topic = ChunkSpec(
        answer="El envío tarda dos días.",
        heading="Envío",
        vector=unit_vector(1),
    )
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, [[on_topic]])
        await session.commit()
        search = HybridKbSearch(session)
        close = await search.search_with_deferred_embed(embedder, site.id, "tiempo de envio")
        far = await search.search_with_deferred_embed(embedder, site.id, "como hornear pan")
    assert [hit.id for hit in close] == [corpus.chunks[0][0].id]
    assert close[0].cosine == pytest.approx(1.0)
    assert close.best_cosine == pytest.approx(1.0)
    assert list(far) == []
    assert far.best_cosine is not None and far.best_cosine < COSINE_FLOOR


async def test_lexical_only_hits_also_carry_a_cosine(migrated_db) -> None:
    # Found by words alone (the query vector points elsewhere); its cosine is still reported.
    chunk = ChunkSpec(
        answer="El envío tarda dos días hábiles.", heading="Envío", vector=unit_vector(1)
    )
    fillers = [
        ChunkSpec(answer=f"Otro tema {n}.", heading=f"Tema {n}", vector=unit_vector(2))
        for n in range(25)
    ]
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, [[*fillers, chunk]])
        await session.commit()
        hits = await HybridKbSearch(session).search(site.id, "dias habiles envio", unit_vector(3))
    found = {hit.id: hit for hit in hits}
    target = found.get(corpus.chunks[0][-1].id)
    assert target is not None
    assert target.cosine is not None


async def test_an_unavailable_embedder_degrades_to_lexical_with_no_cosine(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, [[SHIPPING, RETURNS]])
        await session.commit()
        hits = await HybridKbSearch(session).search_with_deferred_embed(
            UnavailableEmbedder(), site.id, "plazos de envio dos dias"
        )
    assert hits and hits[0].id == corpus.chunks[0][0].id
    assert hits[0].cosine is None
    assert hits.best_cosine is None


async def test_overview_is_a_caller_decision_not_a_keyword_guess(migrated_db) -> None:
    home = ChunkSpec(answer="Somos una tienda de bicicletas.", heading="Quiénes somos")
    other = ChunkSpec(answer="Reparamos frenos y cadenas.", heading="Taller")
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, [[home], [other]])
        await session.commit()
        search = HybridKbSearch(session)
        plain = await search.search(site.id, "what do you guys do")
        overview = await search.search(site.id, "what do you guys do", overview=True)
    assert list(plain) == []
    # The start page leads; a brief start page is supplemented by one passage per other page.
    assert [hit.id for hit in overview] == [corpus.chunks[0][0].id, corpus.chunks[1][0].id]


async def test_only_live_enabled_chunks_are_retrievable(migrated_db) -> None:
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        live = await make_corpus(session, site, [[SHIPPING]])
        staged = await make_corpus(
            session,
            site,
            [[ChunkSpec(answer="El envío tarda dos meses.", heading="Plazos")]],
            start_url="https://tienda.test/staged/",
            state="building",
        )
        await session.commit()
        hits = await _hits(session, site, "tiempo de envio")
        assert [hit.id for hit in hits] == [live.chunks[0][0].id]
        row = await session.scalar(select(KbChunk).where(KbChunk.id == live.chunks[0][0].id))
        row.enabled = False
        await session.commit()
        assert list(await _hits(session, site, "tiempo de envio")) == []
        assert staged.chunks[0][0].id not in {hit.id for hit in await _hits(session, site, "meses")}
