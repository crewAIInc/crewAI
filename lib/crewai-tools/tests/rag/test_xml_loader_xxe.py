"""Tests to verify that XMLLoader is protected against XXE attacks.

defusedxml raises defusedxml.common.EntitiesForbidden when XML contains
external entity declarations, preventing XXE (XML External Entity) attacks
and XML bomb (Billion Laughs) DoS.
"""

from defusedxml.common import EntitiesForbidden

from crewai_tools.rag.loaders.xml_loader import XMLLoader
from crewai_tools.rag.source_content import SourceContent


class TestXMLLoaderXXEProtection:
    """Verify that defusedxml blocks malicious XML payloads."""

    def test_xxe_external_entity_is_blocked(self):
        """An XXE payload referencing an external file must not be parsed."""
        xxe_payload = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE foo ['
            '  <!ENTITY xxe SYSTEM "file:///etc/passwd">'
            ']>'
            '<root>&xxe;</root>'
        )
        loader = XMLLoader()
        sc = SourceContent(xxe_payload)
        # defusedxml will raise EntitiesForbidden, which XMLLoader catches
        # as a ParseError-like exception and falls back to raw content.
        result = loader.load(sc)
        # The content should NOT contain /etc/passwd contents;
        # it should either be the raw XML string (fallback) or raise.
        assert "root:" not in result.content

    def test_xml_bomb_billion_laughs_is_blocked(self):
        """A Billion Laughs (XML bomb) payload must not be expanded."""
        bomb_payload = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE lolz ['
            '  <!ENTITY lol "lol">'
            '  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
            '  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
            ']>'
            '<root>&lol3;</root>'
        )
        loader = XMLLoader()
        sc = SourceContent(bomb_payload)
        result = loader.load(sc)
        # Should not expand into a massive string
        assert len(result.content) < 10000

    def test_safe_xml_still_parses(self):
        """Normal XML without entities should parse correctly."""
        safe_xml = '<root><item>Hello</item><item>World</item></root>'
        loader = XMLLoader()
        sc = SourceContent(safe_xml)
        result = loader.load(sc)
        assert "Hello" in result.content
        assert "World" in result.content
        assert result.metadata["format"] == "xml"
        assert result.metadata["root_tag"] == "root"

    def test_defusedxml_is_used(self):
        """Verify that the XML loader imports from defusedxml, not stdlib xml."""
        import crewai_tools.rag.loaders.xml_loader as mod
        import inspect
        source = inspect.getsource(mod)
        assert "defusedxml" in source
        assert "from xml.etree" not in source
