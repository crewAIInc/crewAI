"""``_chunk_text`` must refuse a configuration that cannot make progress.

``chunk_overlap >= chunk_size`` used to fail in two different ways: ``==`` raised
``range() arg 3 must not be zero`` from inside the comprehension, and ``>``
returned an empty list, silently storing a knowledge source with no content.
"""

import pytest

from crewai.knowledge.source.base_knowledge_source import BaseKnowledgeSource


class _Source(BaseKnowledgeSource):
    def validate_content(self) -> None:
        return None

    def add(self) -> None:
        return None

    async def aadd(self) -> None:
        return None


def test_chunks_step_by_chunk_size_minus_overlap():
    source = _Source(chunk_size=10, chunk_overlap=2)

    assert source._chunk_text("abcdefghijklmnopqrstuvwxyz") == [
        "abcdefghij",
        "ijklmnopqr",
        "qrstuvwxyz",
    ]


def test_empty_text_has_no_chunks():
    assert _Source()._chunk_text("") == []


def test_overlap_equal_to_chunk_size_is_rejected():
    source = _Source(chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError, match="must be smaller than chunk_size"):
        source._chunk_text("x" * 100)


def test_overlap_larger_than_chunk_size_is_rejected():
    source = _Source(chunk_size=10, chunk_overlap=20)

    with pytest.raises(ValueError, match="must be smaller than chunk_size"):
        source._chunk_text("x" * 100)


def test_zero_chunk_size_is_rejected():
    source = _Source(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_size must be greater than 0"):
        source._chunk_text("x" * 100)


def test_negative_overlap_is_rejected():
    source = _Source(chunk_size=10, chunk_overlap=-1)

    with pytest.raises(ValueError, match="must not be negative"):
        source._chunk_text("x" * 100)
