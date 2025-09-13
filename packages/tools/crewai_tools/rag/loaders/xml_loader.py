import os
import xml.etree.ElementTree as ET

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent

class XMLLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs) -> LoaderResult:
        source_ref = source_content.source_ref
        content = source_content.source

        if source_content.is_url():
            content = self._load_from_url(source_ref, kwargs)
        elif os.path.exists(source_ref):
            content = self._load_from_file(source_ref)

        return self._parse_xml(content, source_ref)

    def _load_from_url(self, url: str, kwargs: dict) -> str:
        import requests

        headers = kwargs.get("headers", {
            "Accept": "application/xml, text/xml, text/plain",
            "User-Agent": "Mozilla/5.0 (compatible; crewai-tools XMLLoader)"
        })

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise ValueError(f"Error fetching XML from URL {url}: {str(e)}")

    def _load_from_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as file:
            return file.read()

    def _parse_xml(self, content: str, source_ref: str) -> LoaderResult:
        try:
            if content.strip().startswith('<'):
                root = ET.fromstring(content)
            else:
                root = ET.parse(source_ref).getroot()

            text_parts = []
            for text_content in root.itertext():
                if text_content and text_content.strip():
                    text_parts.append(text_content.strip())

            text = "\n".join(text_parts)
            metadata = {"format": "xml", "root_tag": root.tag}
        except ET.ParseError as e:
            text = content
            metadata = {"format": "xml", "parse_error": str(e)}

        return LoaderResult(
            content=text,
            source=source_ref,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=text)
        )
