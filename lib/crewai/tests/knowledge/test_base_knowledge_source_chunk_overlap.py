"""Test BaseKnowledgeSource rejects a chunk_overlap that would break _chunk_text's slicing step."""

import pytest
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource


def test_chunk_overlap_equal_to_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="chunk_overlap"):
        StringKnowledgeSource(content="hello world", chunk_size=50, chunk_overlap=50)


def test_chunk_overlap_greater_than_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="chunk_overlap"):
        StringKnowledgeSource(content="hello world", chunk_size=50, chunk_overlap=60)


def test_chunk_overlap_smaller_than_chunk_size_is_accepted():
    source = StringKnowledgeSource(content="hello world", chunk_size=50, chunk_overlap=10)
    assert source.chunk_overlap == 10


def test_default_chunk_size_and_overlap_are_valid():
    source = StringKnowledgeSource(content="hello world")
    assert source.chunk_overlap < source.chunk_size
