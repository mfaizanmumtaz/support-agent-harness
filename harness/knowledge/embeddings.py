"""Turning text into vectors: OpenAI for real runs, a hashing fake for tests.

Both sides of search must use the same model: chunks are embedded once when
the knowledge is indexed, and each question is embedded when it's asked.
"""

import hashlib
import math
import os
import re
from typing import Protocol


class Embedder(Protocol):
    @property
    def name(self) -> str: ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    def __init__(self, model: str | None = None) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._model = model or os.environ.get("EMBEDDING_MODEL") or "text-embedding-3-small"
        self._client = OpenAIEmbeddings(model=self._model, max_retries=3)

    @property
    def name(self) -> str:
        return self._model

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._client.aembed_documents(texts) if texts else []

    async def embed_query(self, text: str) -> list[float]:
        return await self._client.aembed_query(text)


class HashEmbedder:
    """No network, but still meaningful: texts sharing words point the same way.

    So offline tests can check that the closest chunk wins.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self._dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hash-{self._dimensions}"

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            slot = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._dimensions
            vector[slot] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
