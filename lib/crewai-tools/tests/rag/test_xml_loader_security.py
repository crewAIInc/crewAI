from crewai_tools.rag.loaders.xml_loader import XMLLoader
from crewai_tools.rag.source_content import SourceContent


class TestXMLLoaderSecurity:
    def test_rejects_internal_entities(self):
        xml_with_entity = """<!DOCTYPE foo [<!ENTITY xxe "EXPANDED">]>
<root>&xxe;</root>"""

        result = XMLLoader().load(SourceContent(xml_with_entity))

        assert result.metadata["format"] == "xml"
        assert "parse_error" in result.metadata
        assert result.content == xml_with_entity
