"""Chunks carry their page context for search; what the bot cites stays untouched."""

from sqlalchemy import select

from app.db import session_maker
from app.models.kb_chunk import KbChunk
from app.services.kb_chunk import pack_chunks
from app.services.kb_extract.markdown import parse_markdown
from app.services.kb_extract.types import EvidenceUnit
from app.services.kb_ingest_chunks import (
    BOILERPLATE_MIN_PAGES,
    dedupe_snapshot_boilerplate,
)
from tests.bot_fixtures import insert_site
from tests.kb_corpus import ChunkSpec, make_corpus

FOOTER = "Todos los pedidos incluyen soporte por correo electrónico durante un año completo."


def _unit(**overrides) -> EvidenceUnit:
    fields = {
        "kind": "section",
        "heading": "Plazos",
        "canonical_question": None,
        "answer_verbatim": "Dos días hábiles.",
        "body_for_search": "Plazos\nDos días hábiles.",
        "display_locator": None,
        "heading_path": ("Envíos", "Nacionales", "Plazos"),
    }
    return EvidenceUnit(**(fields | overrides))


def test_embedded_text_carries_page_title_and_section_path_but_not_the_citation() -> None:
    chunk = pack_chunks(_unit(), page_title="Tienda de bicicletas")[0]
    lines = chunk.body.splitlines()
    assert lines[0] == "Tienda de bicicletas"
    assert lines[1] == "Envíos > Nacionales > Plazos"
    assert chunk.body.endswith("Dos días hábiles.")
    # What the bot quotes is exactly the page passage.
    assert chunk.answer_verbatim == "Dos días hábiles."
    assert chunk.heading == "Plazos"


def test_context_keeps_question_and_aliases_and_drops_repeats() -> None:
    unit = _unit(
        heading="Tienda",
        heading_path=(),
        canonical_question="¿Cuánto tarda?",
        aliases=("plazo", "demora"),
    )
    chunk = pack_chunks(unit, page_title="Tienda")[0]
    assert chunk.body.splitlines()[:3] == ["Tienda", "¿Cuánto tarda?", "plazo | demora"]


def test_without_a_page_title_the_prefix_is_the_section_path() -> None:
    assert pack_chunks(_unit())[0].body.splitlines()[0] == "Envíos > Nacionales > Plazos"


def test_markdown_sections_record_their_ancestor_headings() -> None:
    units = parse_markdown(
        "# Envíos\nIntro del envío.\n## Nacionales\nTexto nacional.\n"
        "### Plazos\nDos días.\n## Internacionales\nTexto fuera.\n",
        "https://tienda.test/envios",
    )
    paths = {unit.heading: unit.heading_path for unit in units}
    assert paths["Plazos"] == ("Envíos", "Nacionales", "Plazos")
    assert paths["Internacionales"] == ("Envíos", "Internacionales")


async def test_a_passage_repeated_on_many_pages_is_kept_once(migrated_db) -> None:
    pages = [
        [ChunkSpec(answer=f"Contenido propio de la página {n}.", heading=f"P{n}")] for n in range(5)
    ]
    for rows in pages[:4]:
        rows.append(ChunkSpec(answer=FOOTER, heading="Pie"))
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, pages)
        await session.commit()
        removed = await dedupe_snapshot_boilerplate(session, corpus.snapshot.id)
        await session.commit()
        remaining = list(
            await session.scalars(select(KbChunk).where(KbChunk.snapshot_id == corpus.snapshot.id))
        )
    assert removed == 3
    assert sum(chunk.answer_verbatim == FOOTER for chunk in remaining) == 1
    own = {chunk.page_id for chunk in remaining if chunk.answer_verbatim.startswith("Contenido")}
    assert own == {page.id for page in corpus.pages}, "no page loses its own content"


async def test_a_passage_on_fewer_pages_than_the_threshold_is_untouched(migrated_db) -> None:
    pages = [
        [ChunkSpec(answer=f"Propio {n} de la página.", heading=f"P{n}"), ChunkSpec(answer=FOOTER)]
        for n in range(BOILERPLATE_MIN_PAGES - 1)
    ]
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, pages)
        await session.commit()
        assert await dedupe_snapshot_boilerplate(session, corpus.snapshot.id) == 0


async def test_a_page_made_only_of_boilerplate_keeps_its_one_chunk(migrated_db) -> None:
    pages = [[ChunkSpec(answer=FOOTER, heading="Pie")] for _ in range(4)]
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, pages)
        await session.commit()
        removed = await dedupe_snapshot_boilerplate(session, corpus.snapshot.id)
        await session.commit()
    # Never empties a page: only the first page's copy is guaranteed, others stay as sole chunks.
    assert removed == 0


async def test_short_repeated_answers_are_not_boilerplate(migrated_db) -> None:
    pages = [
        [ChunkSpec(answer="Sí.", heading=f"Pregunta {n}", question=f"¿Q{n}?")] for n in range(4)
    ]
    for rows in pages:
        rows.append(ChunkSpec(answer="Sí.", heading="Otra", question="¿Otra?"))
    async with session_maker()() as session:
        site = await insert_site(session, "tienda", "Tienda")
        corpus = await make_corpus(session, site, pages)
        await session.commit()
        assert await dedupe_snapshot_boilerplate(session, corpus.snapshot.id) == 0
