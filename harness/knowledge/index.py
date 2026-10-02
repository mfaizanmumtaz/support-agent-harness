"""An in-memory vector index: enough for one business's FAQ, and easy to read.

Production would put the vectors in a vector database (OmniHands uses
Pinecone, one namespace per business). The search contract is the same:
embed the question, score every chunk, drop the weak ones, and keep the best
that fit the token budget.

Embeddings are cached on disk by model and content, so re-running the demo or
the evals doesn't pay to embed the same knowledge again.
"""

import hashlib
import json
import math
from pathlib import Path

from harness.config import RetrievalConfig
from harness.contracts import Hit, SearchResult
from harness.knowledge.chunking import Chunk, chunk_folder
from harness.knowledge.embeddings import Embedder


class KnowledgeIndex:
    def __init__(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        embedder: Embedder,
        config: RetrievalConfig,
    ) -> None:
        self._chunks = chunks
        self._vectors = vectors
        self._embedder = embedder
        self._config = config

    @classmethod
    async def build(
        cls,
        folder: Path,
        embedder: Embedder,
        config: RetrievalConfig | None = None,
        cache_dir: Path | None = None,
    ) -> "KnowledgeIndex":
        config = config or RetrievalConfig()
        chunks = chunk_folder(folder, config.chunk_tokens)
        vectors = await _embed_cached(chunks, embedder, cache_dir)
        return cls(chunks, vectors, embedder, config)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    async def search(self, query: str) -> SearchResult:
        query = query.strip()
        if not query or not self._chunks:
            return SearchResult(query=query, hits=[])
        target = await self._embedder.embed_query(query)
        scored = sorted(
            (
                (_cosine(target, vector), chunk)
                for chunk, vector in zip(self._chunks, self._vectors, strict=True)
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        strong = [(score, chunk) for score, chunk in scored if score >= self._config.min_score]
        hits: list[Hit] = []
        used = 0
        for score, chunk in strong:
            if len(hits) == self._config.top_k:
                break
            # The best chunk always goes in; the rest only while they fit.
            if hits and used + chunk.token_count > self._config.max_context_tokens:
                continue
            hits.append(
                Hit(
                    chunk.id,
                    chunk.source_name,
                    chunk.content,
                    round(score, 4),
                    chunk.token_count,
                    chunk.heading,
                )
            )
            used += chunk.token_count
        return SearchResult(
            query=query,
            hits=hits,
            candidates=len(scored),
            below_threshold=len(scored) - len(strong),
        )


async def _embed_cached(
    chunks: list[Chunk], embedder: Embedder, cache_dir: Path | None
) -> list[list[float]]:
    if cache_dir is None:
        return await embedder.embed_documents([chunk.content for chunk in chunks])
    path = cache_dir / f"embeddings-{_slug(embedder.name)}.json"
    cached = _read_cache(path)
    keys = [hashlib.sha256(chunk.content.encode()).hexdigest() for chunk in chunks]
    missing = [(key, chunk) for key, chunk in zip(keys, chunks, strict=True) if key not in cached]
    if missing:
        vectors = await embedder.embed_documents([chunk.content for _, chunk in missing])
        cached.update({key: vector for (key, _), vector in zip(missing, vectors, strict=True)})
        _write_cache(path, cached)
    return [cached[key] for key in keys]


def _read_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    data: dict[str, list[float]] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _write_cache(path: Path, cached: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cached), encoding="utf-8")


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0


def _slug(name: str) -> str:
    return "".join(char if char.isalnum() or char in "-." else "_" for char in name)
