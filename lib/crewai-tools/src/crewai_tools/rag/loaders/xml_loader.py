from typing import Any
from xml.etree.ElementTree import ParseError, fromstring, parse

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.loaders.utils import load_from_url
from crewai_tools.rag.source_content import SourceContent


class XMLLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs: Any) -> LoaderResult:  # type: ignore[override]
        """Load and parse XML content from various sources.

        Args:
            source_content: SourceContent: The source content to load.
            **kwargs: Additional keyword arguments for loading from URL.

        Returns:
            LoaderResult: The result of loading and parsing the XML content.
        """
        source_ref = source_content.source_ref
        content = source_content.source

        if source_content.is_url():
            content = load_from_url(
                source_ref,
                kwargs,
                accept_header="application/xml, text/xml, text/plain",
                loader_name="XMLLoader",
            )
        elif source_content.path_exists():
            content = self._load_from_file(source_ref)

        return self._parse_xml(content, source_ref)

    @staticmethod
    def _load_from_file(path: str) -> str:
        with open(path, encoding="utf-8") as file:
            return file.read()

    def _parse_xml(self, content: str, source_ref: str) -> LoaderResult:
        try:
            if content.strip().startswith("<"):
                root = fromstring(content)  # noqa: S314
            else:
                root = parse(source_ref).getroot()  # noqa: S314

            text_parts = []
            for text_content in root.itertext():
                if text_content and text_content.strip():
                    text_parts.append(text_content.strip())  # noqa: PERF401

            text = "\n".join(text_parts)
            metadata = {"format": "xml", "root_tag": root.tag}
        except ParseError as e:
            text = content
            metadata = {"format": "xml", "parse_error": str(e)}

        return LoaderResult(
            content=text,
            source=source_ref,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=text),
        )
