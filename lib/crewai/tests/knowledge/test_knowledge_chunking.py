"""Chunking shared by the built-in knowledge sources."""

import pytest

from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


def _chunks(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    source = StringKnowledgeSource(
        content=text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return source._chunk_text(text)


def test_text_shorter_than_a_chunk_is_stored_once() -> None:
    text = "x" * 3900

    assert _chunks(text, chunk_size=4000, chunk_overlap=200) == [text]


def test_last_chunk_is_not_the_tail_of_the_previous_one() -> None:
    text = "abcdefghijklmnopqrstuvwxyz" * 2

    assert _chunks(text[:48], chunk_size=20, chunk_overlap=5) == [
        text[0:20],
        text[15:35],
        text[30:48],
    ]


def test_empty_text_has_no_chunks() -> None:
    assert _chunks("", chunk_size=10, chunk_overlap=3) == []


@pytest.mark.parametrize("chunk_overlap", [0, 3, 9])
def test_chunks_cover_the_text_and_only_the_last_reaches_its_end(
    chunk_overlap: int,
) -> None:
    chunk_size = 10
    step = chunk_size - chunk_overlap
    for length in range(1, 60):
        text = "".join(chr(ord("a") + i % 26) for i in range(length))

        chunks = _chunks(text, chunk_size, chunk_overlap)

        assert "".join(c[:step] for c in chunks[:-1]) + chunks[-1] == text
        assert all(
            i * step + len(chunk) < length for i, chunk in enumerate(chunks[:-1])
        )
