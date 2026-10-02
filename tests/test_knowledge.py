"""Chunking and search, with the offline hashing embedder."""

from pathlib import Path

from harness.config import RetrievalConfig
from harness.knowledge import HashEmbedder, KnowledgeIndex, chunk_document
from harness.knowledge.chunking import sections_of
from harness.sample import knowledge_folder

FAQ = """# Policies

Intro line.

## Pets
Dogs are welcome for €15 per night.

## Smoking
All rooms are non-smoking.

## Empty heading
"""


def test_each_heading_is_a_section_and_empty_ones_are_dropped() -> None:
    sections = sections_of(FAQ)

    assert [section.heading for section in sections] == ["Policies", "Pets", "Smoking"]
    assert sections[1].text == "## Pets\nDogs are welcome for €15 per night."


def test_small_neighbouring_sections_are_merged_up_to_the_chunk_size() -> None:
    chunks = chunk_document("policies", FAQ, chunk_tokens=400)

    assert len(chunks) == 1
    assert chunks[0].heading == "Policies"
    assert "Dogs are welcome" in chunks[0].content and "non-smoking" in chunks[0].content


def test_a_section_too_big_for_one_chunk_is_cut_at_paragraphs() -> None:
    long_section = "## Long\n" + "\n\n".join(f"Paragraph {n} " + "word " * 40 for n in range(6))

    chunks = chunk_document("doc", long_section, chunk_tokens=100)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 100 for chunk in chunks)
    assert all(chunk.heading == "Long" for chunk in chunks)


async def test_the_closest_chunk_wins_and_weak_ones_are_dropped() -> None:
    index = await KnowledgeIndex.build(
        knowledge_folder(), HashEmbedder(), RetrievalConfig(chunk_tokens=120, min_score=0.1)
    )

    result = await index.search("are dogs allowed pets")

    assert result.hits, "expected a hit"
    assert "Dogs are welcome" in result.hits[0].content
    assert result.candidates == len(index.chunks)
    assert all(hit.score >= 0.1 for hit in result.hits)


async def test_the_shown_chunks_stay_within_the_token_budget() -> None:
    config = RetrievalConfig(chunk_tokens=120, min_score=-1, top_k=10, max_context_tokens=200)
    index = await KnowledgeIndex.build(knowledge_folder(), HashEmbedder(), config)

    result = await index.search("breakfast")

    assert sum(hit.token_count for hit in result.hits) <= 200 or len(result.hits) == 1


async def test_embeddings_are_cached_by_content(tmp_path: Path) -> None:
    class CountingEmbedder(HashEmbedder):
        embedded = 0

        async def embed_documents(self, texts: list[str]) -> list[list[float]]:
            CountingEmbedder.embedded += len(texts)
            return await super().embed_documents(texts)

    first = await KnowledgeIndex.build(knowledge_folder(), CountingEmbedder(), cache_dir=tmp_path)
    count = CountingEmbedder.embedded
    await KnowledgeIndex.build(knowledge_folder(), CountingEmbedder(), cache_dir=tmp_path)

    assert count == len(first.chunks)
    assert CountingEmbedder.embedded == count
