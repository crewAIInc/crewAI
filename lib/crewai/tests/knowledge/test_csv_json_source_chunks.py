"""CSV and JSON knowledge sources chunk the file text, not the dict repr."""

from unittest.mock import AsyncMock, patch

import pytest

from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.json_knowledge_source import JSONKnowledgeSource


def test_csv_source_chunks_the_file_text(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("name,score\nann,1\n", encoding="utf-8")
    source = CSVKnowledgeSource(file_paths=[path])

    with patch.object(CSVKnowledgeSource, "_save_documents") as save:
        source.add()

    assert source.chunks == ["name score\nann 1\n"]
    assert "WindowsPath" not in source.chunks[0]
    save.assert_called_once()


def test_csv_source_chunks_each_file_separately(tmp_path):
    first = tmp_path / "first.csv"
    first.write_text("a,1\n", encoding="utf-8")
    second = tmp_path / "second.csv"
    second.write_text("b,2\n", encoding="utf-8")
    source = CSVKnowledgeSource(file_paths=[first, second])

    with patch.object(CSVKnowledgeSource, "_save_documents"):
        source.add()

    assert source.chunks == ["a 1\n", "b 2\n"]


@pytest.mark.asyncio
async def test_csv_source_async_chunks_the_file_text(tmp_path):
    path = tmp_path / "scores.csv"
    path.write_text("name,score\nann,1\n", encoding="utf-8")
    source = CSVKnowledgeSource(file_paths=[path])

    with patch.object(
        CSVKnowledgeSource, "_asave_documents", new=AsyncMock()
    ) as save:
        await source.aadd()

    assert source.chunks == ["name score\nann 1\n"]
    save.assert_awaited_once()


def test_json_source_chunks_the_file_text(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text('{"name": "ann", "score": 1}', encoding="utf-8")
    source = JSONKnowledgeSource(file_paths=[path])

    with patch.object(JSONKnowledgeSource, "_save_documents") as save:
        source.add()

    assert "name: ann" in source.chunks[0]
    assert "score: 1" in source.chunks[0]
    assert "WindowsPath" not in source.chunks[0]
    save.assert_called_once()
