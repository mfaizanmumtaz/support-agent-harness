"""Markdown in, chunks out, keeping the document's own structure.

FAQ pages and policies are many short sections ("## Can I bring a pet?" and
two lines). Cutting each alone gives chunks too small to carry context;
cutting by size ignores where one topic ends. So whole sections stay together,
small neighbours are joined up to `chunk_tokens`, and only a section too big by
itself is cut, at paragraph breaks.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from harness.tokens import count_tokens

_HEADING = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Section:
    heading: str
    text: str


@dataclass(frozen=True)
class Chunk:
    id: str
    source_name: str
    heading: str
    content: str
    token_count: int


def sections_of(markdown: str) -> list[Section]:
    """One section per heading, its heading line included so a chunk says what it's about."""
    marks = list(_HEADING.finditer(markdown))
    sections: list[Section] = []
    intro = markdown[: marks[0].start()].strip() if marks else markdown.strip()
    if intro:
        sections.append(Section("", intro))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(markdown)
        text = markdown[mark.start() : end].strip()
        if text.count("\n"):  # a heading with nothing under it adds nothing
            sections.append(Section(mark.group(2), text))
    return sections


def chunk_document(source_name: str, markdown: str, chunk_tokens: int) -> list[Chunk]:
    pieces: list[tuple[str, str]] = []
    group: list[Section] = []
    size = 0

    def flush() -> None:
        nonlocal group, size
        if group:
            # A merged chunk keeps the first heading: it is where the reader lands.
            pieces.append((group[0].heading, "\n\n".join(section.text for section in group)))
        group, size = [], 0

    for section in sections_of(markdown):
        tokens = count_tokens(section.text)
        if tokens > chunk_tokens:
            flush()
            pieces.extend((section.heading, part) for part in _split(section.text, chunk_tokens))
            continue
        if group and size + tokens > chunk_tokens:
            flush()
        group.append(section)
        size += tokens
    flush()
    return [
        Chunk(f"{source_name}#{index}", source_name, heading, text, count_tokens(text))
        for index, (heading, text) in enumerate(pieces)
    ]


def chunk_folder(folder: Path, chunk_tokens: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(folder.glob("*.md")):
        chunks.extend(chunk_document(path.stem, path.read_text(encoding="utf-8"), chunk_tokens))
    return chunks


def _split(text: str, chunk_tokens: int) -> list[str]:
    """A section too big for one chunk, cut at paragraph breaks."""
    parts: list[str] = []
    current: list[str] = []
    for paragraph in text.split("\n\n"):
        if current and count_tokens("\n\n".join([*current, paragraph])) > chunk_tokens:
            parts.append("\n\n".join(current))
            current = []
        current.append(paragraph)
    if current:
        parts.append("\n\n".join(current))
    return parts
