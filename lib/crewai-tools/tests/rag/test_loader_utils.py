from unittest.mock import patch

from crewai_tools.rag.loaders.json_loader import JSONLoader
from crewai_tools.rag.loaders.xml_loader import XMLLoader
from crewai_tools.rag.source_content import SourceContent
import pytest
from requests import Response
from requests.utils import get_encoding_from_headers


@pytest.mark.parametrize("charset", ["", "; charset=utf-8-sig"])
@pytest.mark.parametrize(
    "loader, content_type, content, metadata",
    [
        (
            JSONLoader,
            "application/octet-stream",
            '{"message": "hello"}',
            {"format": "json", "type": "dict", "size": 1},
        ),
        (
            XMLLoader,
            "application/xml",
            "<root>hello</root>",
            {"format": "xml", "root_tag": "root"},
        ),
    ],
)
def test_url_loaders_preserve_bom_decoding(
    loader: type[JSONLoader] | type[XMLLoader],
    content_type: str,
    content: str,
    metadata: dict[str, str | int],
    charset: str,
) -> None:
    """Keep JSON and XML parsing compatible with automatic or explicit BOM decoding."""
    response = Response()
    response.status_code = 200
    response.headers["Content-Type"] = content_type + charset
    response.encoding = get_encoding_from_headers(response.headers)
    response._content = content.encode("utf-8-sig")
    with patch("crewai_tools.security.safe_requests._raw_get", return_value=response):
        result = loader().load(SourceContent("https://example.com/data"))

    assert result.metadata == metadata
    assert "hello" in result.content
