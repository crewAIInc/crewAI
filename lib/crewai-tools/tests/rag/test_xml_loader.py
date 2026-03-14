import os
import tempfile
from unittest.mock import patch

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.xml_loader import XMLLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


class TestXMLLoader:
    def test_parse_valid_xml_string(self):
        loader = XMLLoader()
        source = SourceContent("<root><item>Hello</item><item>World</item></root>")
        result = loader.load(source)

        assert isinstance(result, LoaderResult)
        assert "Hello" in result.content
        assert "World" in result.content
        assert result.metadata["format"] == "xml"
        assert result.metadata["root_tag"] == "root"

    def test_parse_xml_from_file(self):
        xml_content = "<root><title>Test Title</title><body>Test Body</body></root>"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            loader = XMLLoader()
            source = SourceContent(temp_path)
            result = loader.load(source)

            assert isinstance(result, LoaderResult)
            assert "Test Title" in result.content
            assert "Test Body" in result.content
            assert result.metadata["format"] == "xml"
            assert result.metadata["root_tag"] == "root"
        finally:
            os.unlink(temp_path)

    def test_parse_invalid_xml_returns_raw_content(self):
        loader = XMLLoader()
        invalid_xml = "<root><unclosed>"
        source = SourceContent(invalid_xml)
        result = loader.load(source)

        assert isinstance(result, LoaderResult)
        assert result.content == invalid_xml
        assert "parse_error" in result.metadata
        assert result.metadata["format"] == "xml"

    def test_parse_nested_xml(self):
        loader = XMLLoader()
        xml = (
            "<root>"
            "<parent><child>Nested text</child></parent>"
            "<sibling>Other text</sibling>"
            "</root>"
        )
        source = SourceContent(xml)
        result = loader.load(source)

        assert "Nested text" in result.content
        assert "Other text" in result.content

    def test_parse_xml_with_attributes(self):
        loader = XMLLoader()
        xml = '<root><item id="1">First</item><item id="2">Second</item></root>'
        source = SourceContent(xml)
        result = loader.load(source)

        assert "First" in result.content
        assert "Second" in result.content
        assert result.metadata["root_tag"] == "root"

    def test_parse_empty_xml_elements(self):
        loader = XMLLoader()
        xml = "<root><empty></empty><filled>Content</filled></root>"
        source = SourceContent(xml)
        result = loader.load(source)

        assert "Content" in result.content
        assert result.metadata["format"] == "xml"

    def test_doc_id_consistency(self):
        loader = XMLLoader()
        xml = "<root><item>Consistent</item></root>"
        source = SourceContent(xml)
        result1 = loader.load(source)
        result2 = loader.load(source)

        assert result1.doc_id == result2.doc_id

    @patch("crewai_tools.rag.loaders.xml_loader.load_from_url")
    def test_load_from_url(self, mock_load_url):
        mock_load_url.return_value = "<root><data>URL content</data></root>"
        loader = XMLLoader()
        source = SourceContent("https://example.com/data.xml")
        result = loader.load(source)

        assert isinstance(result, LoaderResult)
        assert "URL content" in result.content
        mock_load_url.assert_called_once()

    def test_xxe_entity_expansion_blocked(self):
        """Test that XML External Entity (XXE) attacks are blocked by defusedxml."""
        loader = XMLLoader()
        xxe_payload = (
            '<?xml version="1.0"?>'
            "<!DOCTYPE foo ["
            '  <!ENTITY xxe SYSTEM "file:///etc/passwd">'
            "]>"
            "<root>&xxe;</root>"
        )
        source = SourceContent(xxe_payload)
        result = loader.load(source)

        # defusedxml should block the entity expansion and raise EntitiesForbidden,
        # which is a subclass of ParseError, so _parse_xml catches it and returns
        # raw content with a parse_error in metadata.
        assert "parse_error" in result.metadata
        assert result.metadata["format"] == "xml"
        # The raw payload should NOT have resolved the entity
        assert "/etc/passwd" not in result.content or result.content == xxe_payload

    def test_xxe_billion_laughs_blocked(self):
        """Test that XML bomb (Billion Laughs) attacks are blocked by defusedxml."""
        loader = XMLLoader()
        billion_laughs = (
            '<?xml version="1.0"?>'
            "<!DOCTYPE lolz ["
            '  <!ENTITY lol "lol">'
            '  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
            '  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">'
            "]>"
            "<root>&lol3;</root>"
        )
        source = SourceContent(billion_laughs)
        result = loader.load(source)

        # defusedxml blocks entity expansion, resulting in a parse error
        assert "parse_error" in result.metadata
        assert result.metadata["format"] == "xml"

    def test_xxe_parameter_entity_blocked(self):
        """Test that parameter entity attacks are blocked by defusedxml."""
        loader = XMLLoader()
        xxe_param = (
            '<?xml version="1.0"?>'
            "<!DOCTYPE foo ["
            '  <!ENTITY % xxe SYSTEM "file:///etc/passwd">'
            "  %xxe;"
            "]>"
            "<root>test</root>"
        )
        source = SourceContent(xxe_param)
        result = loader.load(source)

        assert "parse_error" in result.metadata

    def test_xxe_file_from_file_blocked(self):
        """Test that XXE attacks via file loading are also blocked."""
        xxe_content = (
            '<?xml version="1.0"?>'
            "<!DOCTYPE foo ["
            '  <!ENTITY xxe SYSTEM "file:///etc/passwd">'
            "]>"
            "<root>&xxe;</root>"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        ) as f:
            f.write(xxe_content)
            temp_path = f.name

        try:
            loader = XMLLoader()
            source = SourceContent(temp_path)
            result = loader.load(source)

            assert "parse_error" in result.metadata
            assert result.metadata["format"] == "xml"
        finally:
            os.unlink(temp_path)

    def test_defusedxml_is_used_not_stdlib(self):
        """Verify that the XMLLoader imports from defusedxml, not xml.etree.ElementTree."""
        import crewai_tools.rag.loaders.xml_loader as xml_loader_module

        assert "defusedxml" in xml_loader_module.fromstring.__module__

    def test_arxiv_tool_uses_defusedxml(self):
        """Verify that ArxivPaperTool imports from defusedxml, not xml.etree.ElementTree."""
        import crewai_tools.tools.arxiv_paper_tool.arxiv_paper_tool as arxiv_module

        ET = arxiv_module.ET
        assert "defusedxml" in ET.__name__
