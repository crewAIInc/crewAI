"""Tests for XML loader inputs resolved from text, disk, and HTTP."""

from pathlib import Path
from unittest.mock import patch

import pytest

from crewai_tools.rag.loaders.xml_loader import XMLLoader
from crewai_tools.rag.source_content import SourceContent


@pytest.mark.parametrize("kind", ["inline", "file", "url"])
@pytest.mark.parametrize("prefix", ["", "\ufeff"])
def test_load_resolved_xml(kind: str, prefix: str, tmp_path: Path) -> None:
    content = f"{prefix}<lesson><title>Study plan</title></lesson>"
    if kind == "file":
        path = tmp_path / "lesson.xml"
        path.write_text(content, encoding="utf-8")
        source = SourceContent(path)
    elif kind == "url":
        source = SourceContent("https://example.invalid/lesson.xml")
    else:
        source = SourceContent(content)

    with patch(
        "crewai_tools.rag.loaders.xml_loader.load_from_url", return_value=content
    ) as download:
        result = XMLLoader().load(source)

    assert download.call_count == (1 if kind == "url" else 0)
    assert result.content == "Study plan"
    assert result.source == source.source_ref
    assert result.metadata == {"format": "xml", "root_tag": "lesson"}


@pytest.mark.parametrize("content", ["", "not XML", "<unclosed>"])
def test_invalid_inline_xml_returns_raw_text(content: str) -> None:
    source = SourceContent(content)
    result = XMLLoader().load(source)

    assert result.content == content
    assert result.source == source.source_ref
    assert result.metadata["format"] == "xml"
    assert result.metadata["parse_error"]
