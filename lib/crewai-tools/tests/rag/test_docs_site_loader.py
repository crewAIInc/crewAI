from unittest.mock import patch

from crewai_tools.rag.base_loader import BaseLoader
from crewai_tools.rag.loaders.docs_site_loader import DocsSiteLoader
from crewai_tools.rag.source_content import SourceContent
import pytest
from requests import Response


@pytest.mark.parametrize(
    "docs_url",
    ["https://docs.example.com/guide", "https://example.com/docs/guide?version=2"],
)
def test_load_preserves_document_source(docs_url: str) -> None:
    """Keep the original URL available for document attribution and replacement."""
    response = Response()
    response.status_code = 200
    response.url = docs_url
    response.encoding = "utf-8"
    response._content = (
        b"<html><title>Guide</title><main><h1>Getting started</h1>"
        b"<p>Install the package.</p></main></html>"
    )

    with patch(
        "crewai_tools.rag.loaders.docs_site_loader.safe_get", return_value=response
    ) as mock_get:
        result = DocsSiteLoader().load(SourceContent(docs_url))

    mock_get.assert_called_once_with(docs_url, timeout=30)
    assert "Install the package." in result.content
    assert result.source == docs_url
    assert result.metadata["source"] == docs_url
    assert result.doc_id == BaseLoader.generate_doc_id(docs_url, result.content)
