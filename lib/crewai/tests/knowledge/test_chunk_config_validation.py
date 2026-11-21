"""Tests that knowledge source chunk configurations are validated."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource

LONG_TEXT = "A" * 500


def test_rejects_overlap_equal_to_size():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        StringKnowledgeSource(
            content=LONG_TEXT, chunk_size=100, chunk_overlap=100
        )


def test_rejects_overlap_greater_than_size():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        StringKnowledgeSource(
            content=LONG_TEXT, chunk_size=100, chunk_overlap=150
        )


def test_rejects_negative_overlap():
    with pytest.raises(ValidationError, match="chunk_overlap"):
        StringKnowledgeSource(content=LONG_TEXT, chunk_size=100, chunk_overlap=-10)


def test_rejects_non_positive_chunk_size():
    with pytest.raises(ValidationError, match="chunk_size"):
        StringKnowledgeSource(content=LONG_TEXT, chunk_size=0, chunk_overlap=0)


def test_validation_is_inherited_by_file_sources(tmp_path: Path):
    text_file = tmp_path / "notes.txt"
    text_file.write_text(LONG_TEXT, encoding="utf-8")

    with pytest.raises(ValidationError, match="chunk_overlap"):
        TextFileKnowledgeSource(
            file_paths=[text_file], chunk_size=100, chunk_overlap=100
        )


def test_valid_configuration_still_chunks_with_overlap():
    source = StringKnowledgeSource(
        content=LONG_TEXT, chunk_size=100, chunk_overlap=20
    )
    chunks = source._chunk_text(LONG_TEXT)

    assert len(chunks) == 7
    assert all(len(c) == 100 for c in chunks[:-1])
    assert chunks[-1] == LONG_TEXT[480:]
    # consecutive windows overlap by chunk_overlap characters
    assert all(
        chunks[i][80:] == chunks[i + 1][:20] for i in range(len(chunks) - 1)
    )
