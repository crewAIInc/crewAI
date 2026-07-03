"""Regression tests for XMLLoader against XXE and entity-expansion attacks."""

from unittest.mock import patch

import pytest
from defusedxml.common import EntitiesForbidden

from crewai_tools.rag.loaders.xml_loader import XMLLoader
from crewai_tools.rag.source_content import SourceContent


class TestXMLLoaderSecurity:
    """Ensure XMLLoader rejects XML documents that would trigger XXE or entity
    expansion attacks when parsed with the standard library.
    """

    def test_rejects_external_entity_expansion(self):
        """A document referencing an external entity must not resolve it.

        Using stdlib xml.etree, the entity would either be resolved (leaking
        file contents) or silently dropped depending on the parser. defusedxml
        raises instead, which is the desired behavior.
        """
        malicious = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE root ['
            '<!ENTITY xxe SYSTEM "file:///etc/passwd">'
            ']>'
            '<root>&xxe;</root>'
        )

        loader = XMLLoader()
        source = SourceContent(malicious)

        with pytest.raises(EntitiesForbidden):
            loader.load(source)

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
        source = SourceContent(malicious)

        with pytest.raises(EntitiesForbidden):
            loader.load(source)

    def test_parses_safe_xml(self):
        """A benign XML document without a DOCTYPE must still parse."""
        safe = '<?xml version="1.0"?><root><item>hello</item><item>world</item></root>'

        loader = XMLLoader()
        result = loader.load(SourceContent(safe))

        assert "hello" in result.content
        assert "world" in result.content
        assert result.metadata["format"] == "xml"
        assert result.metadata["root_tag"] == "root"

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
        source = SourceContent("https://example.com/feed.xml")

        with pytest.raises(EntitiesForbidden):
            loader.load(source)
