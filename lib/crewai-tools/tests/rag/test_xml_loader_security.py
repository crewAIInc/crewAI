"""Regression tests for XMLLoader against XXE and entity-expansion attacks."""

from unittest.mock import patch

from crewai_tools.rag.loaders.xml_loader import XMLLoader
from crewai_tools.rag.source_content import SourceContent


class TestXMLLoaderSecurity:
    """Ensure XMLLoader rejects XML documents that would trigger XXE or entity
    expansion attacks when parsed with the standard library.

    The loader keeps its "return a LoaderResult with an error in metadata"
    contract for malformed input, so malicious documents are contained at the
    loader boundary rather than aborting the caller (e.g. RAG.add).
    """

    def test_rejects_external_entity_expansion(self):
        """A document referencing an external entity must not resolve it.

        The stdlib xml.etree parser would resolve `SYSTEM` entities (leaking
        file contents). defusedxml refuses; XMLLoader captures that and
        surfaces `security_error` in metadata.
        """
        malicious = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE root ['
            '<!ENTITY xxe SYSTEM "file:///etc/passwd">'
            ']>'
            '<root>&xxe;</root>'
        )

        loader = XMLLoader()
        result = loader.load(SourceContent(malicious))

        assert result.content == ""
        assert "security_error" in result.metadata
        assert result.metadata["format"] == "xml"
        assert "root_tag" not in result.metadata

    def test_rejects_billion_laughs(self):
        """Nested entity expansion (billion laughs) must be refused."""
        malicious = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE lolz ['
            '<!ENTITY lol "lol">'
            '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
            '<!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
            ']>'
            '<lolz>&lol3;</lolz>'
        )

        loader = XMLLoader()
        result = loader.load(SourceContent(malicious))

        assert result.content == ""
        assert "security_error" in result.metadata
        assert result.metadata["format"] == "xml"

    def test_parses_safe_xml(self):
        """A benign XML document without a DOCTYPE must still parse."""
        safe = '<?xml version="1.0"?><root><item>hello</item><item>world</item></root>'

        loader = XMLLoader()
        result = loader.load(SourceContent(safe))

        assert "hello" in result.content
        assert "world" in result.content
        assert result.metadata["format"] == "xml"
        assert result.metadata["root_tag"] == "root"
        assert "security_error" not in result.metadata

    def test_malformed_xml_still_returns_parse_error(self):
        """Non-security parse failures still land in `parse_error`, not
        `security_error`, so existing consumers keep working.
        """
        broken = "<root><unclosed>"

        loader = XMLLoader()
        result = loader.load(SourceContent(broken))

        assert "parse_error" in result.metadata
        assert "security_error" not in result.metadata

    @patch("crewai_tools.rag.loaders.xml_loader.load_from_url")
    def test_rejects_xxe_from_url_source(self, mock_load):
        """XML fetched from a remote URL is still validated."""
        mock_load.return_value = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE root ['
            '<!ENTITY xxe SYSTEM "file:///etc/passwd">'
            ']>'
            '<root>&xxe;</root>'
        )

        loader = XMLLoader()
        result = loader.load(SourceContent("https://example.com/feed.xml"))

        assert result.content == ""
        assert "security_error" in result.metadata
