"""Test doubles for the embedding provider.

Production code has no fake embedder and no fallback to one: a process without
``OPENAI_API_KEY`` refuses to start. Tests inject these explicitly, either by passing
one to the service under test or through the autouse ``_embedding_provider`` fixture in
``conftest.py``.
"""

from collections.abc import Mapping
from hashlib import sha256

from app.settings import get_settings


def unit_vector(axis: int, dim: int | None = None) -> list[float]:
    size = get_settings().openai_embed_dim if dim is None else dim
    vector = [0.0] * size
    if 0 <= axis < size:
        vector[axis] = 1.0
    return vector


def axis_for_text(text: str) -> int:
    """Fixture topics for the shared test corpus; anything else lands on axis 10."""
    lowered = text.casefold()
    if any(token in lowered for token in ("24-48", "how fast", "turnaround", "how long")):
        return 0
    if "fcra" in lowered or "fair credit" in lowered:
        return 1
    if any(token in lowered for token in ("pricing", "how much", "cost", "price")):
        return 2
    if "dot" in lowered:
        return 3
    if "portal" in lowered:
        return 4
    return 10


class FakeEmbedder:
    embedder_id = "fake:axes"

    @property
    def dim(self) -> int:
        return get_settings().openai_embed_dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [unit_vector(axis_for_text(text)) for text in texts]

    async def embed_query(self, text: str) -> list[float] | None:
        return unit_vector(axis_for_text(text))


class MappedEmbedder:
    """Maps chosen substrings to chosen axes, so a test states exactly what is 'close'.

    The first substring found (case-insensitive, in mapping order) decides the axis.
    Text matching none gets its own axis derived from the text, so unrelated texts
    never look close to each other; pass ``default_axis`` to put them all on one axis.
    """

    embedder_id = "fake:mapped"

    def __init__(self, axes: Mapping[str, int], default_axis: int | None = None) -> None:
        self._axes = {key.casefold(): axis for key, axis in axes.items()}
        self._default = default_axis

    @property
    def dim(self) -> int:
        return get_settings().openai_embed_dim

    def axis(self, text: str) -> int:
        lowered = text.casefold()
        for key, axis in self._axes.items():
            if key in lowered:
                return axis
        if self._default is not None:
            return self._default
        return 200 + int(sha256(lowered.encode()).hexdigest(), 16) % 1000

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [unit_vector(self.axis(text)) for text in texts]

    async def embed_query(self, text: str) -> list[float] | None:
        return unit_vector(self.axis(text))


class UnavailableEmbedder:
    """An embedding provider that is down: queries return nothing, documents raise."""

    embedder_id = "fake:down"

    @property
    def dim(self) -> int:
        return get_settings().openai_embed_dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise TimeoutError

    async def embed_query(self, text: str) -> list[float] | None:
        raise TimeoutError


class FakeOpenAIEmbedder(FakeEmbedder):
    """Stands in for ``OpenAIEmbedder`` where ``default_embedder()`` builds one."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    @classmethod
    async def close_shared_clients(cls) -> None:
        return None
