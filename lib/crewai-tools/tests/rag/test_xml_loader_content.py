from pathlib import Path
from unittest.mock import patch

import pytest
import requests

from crewai_tools.rag.loaders.xml_loader import XMLLoader
from crewai_tools.rag.source_content import SourceContent


@pytest.mark.parametrize("source_kind", ["inline", "file", "url"])
@pytest.mark.parametrize("prefix", ["", "\ufeff"])
def test_xml_content_is_parsed_without_reopening_its_reference(
    source_kind: str, prefix: str, tmp_path: Path
) -> None:
    """Parse plain and BOM-prefixed XML from inline, local and URL sources."""
    content = prefix + "<catalog><item>First</item><item>Second</item></catalog>"
    source = content
    if source_kind == "file":
        path = tmp_path / "catalog.xml"
        path.write_text(content, encoding="utf-8")
        source = str(path)
    elif source_kind == "url":
        source = "https://example.com/catalog.xml"

    response = requests.Response()
    response.status_code = 200
    response.encoding = "utf-8"
    response._content = content.encode("utf-8")
    with patch("crewai_tools.security.safe_requests.safe_get", return_value=response):
        source_content = SourceContent(source)
        result = XMLLoader().load(source_content)

    assert result.content == "First\nSecond"
    assert result.metadata == {"format": "xml", "root_tag": "catalog"}
    assert result.source == source_content.source_ref
    assert result.doc_id


@pytest.mark.parametrize("content", ["", "not XML", "<root>unclosed"])
def test_malformed_inline_xml_returns_parse_error_metadata(content: str) -> None:
    """Preserve malformed input instead of trying to open its source hash."""
    result = XMLLoader().load(SourceContent(content))

    assert result.content == content
    assert result.metadata["format"] == "xml"
    assert result.metadata["parse_error"]
