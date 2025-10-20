import re
from typing import Final

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.loaders.utils import load_from_url
from crewai_tools.rag.source_content import SourceContent


_IMPORT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^import\s+.*?\n", re.MULTILINE)
_EXPORT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^export\s+.*?(?:\n|$)", re.MULTILINE
)
_JSX_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
_EXTRA_NEWLINES_PATTERN: Final[re.Pattern[str]] = re.compile(r"\n\s*\n\s*\n")


class MDXLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        source_ref = source_content.source_ref
        content = source_content.source

        if source_content.is_url():
            content = load_from_url(
                source_ref,
                kwargs,
                accept_header="text/markdown, text/x-markdown, text/plain",
                loader_name="MDXLoader",
            )
        elif source_content.path_exists():
            content = self._load_from_file(source_ref)

        return self._parse_mdx(content, source_ref)

    @staticmethod
    def _load_from_file(path: str) -> str:
        with open(path, encoding="utf-8") as file:
            return file.read()

    def _parse_mdx(self, content: str, source_ref: str) -> LoaderResult:
        cleaned_content = content

        # Remove import statements
        cleaned_content = _IMPORT_PATTERN.sub("", cleaned_content)

        # Remove export statements
        cleaned_content = _EXPORT_PATTERN.sub("", cleaned_content)

        # Remove JSX tags (simple approach)
        cleaned_content = _JSX_TAG_PATTERN.sub("", cleaned_content)

        # Clean up extra whitespace
        cleaned_content = _EXTRA_NEWLINES_PATTERN.sub("\n\n", cleaned_content)
        cleaned_content = cleaned_content.strip()

        metadata = {"format": "mdx"}
        return LoaderResult(
            content=cleaned_content,
            source=source_ref,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=cleaned_content),
        )
