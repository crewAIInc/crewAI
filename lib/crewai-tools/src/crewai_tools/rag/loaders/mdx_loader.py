import re
from typing import Any, Final

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.loaders.utils import load_from_url
from crewai_tools.rag.source_content import SourceContent


_IMPORT_PATTERN: Final[re.Pattern[str]] = re.compile(r"^import\s+.*?\n", re.MULTILINE)
_EXPORT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^export\s+.*?(?:\n|$)", re.MULTILINE
)
_JSX_TAG_PATTERN: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")
_EXTRA_NEWLINES_PATTERN: Final[re.Pattern[str]] = re.compile(r"\n\s*\n\s*\n")

# Literal regions whose text must be preserved verbatim: fenced code blocks and
# inline code spans. MDX cleanup (imports/exports, JSX tags, blank-line
# collapsing) must not run inside them, or it corrupts the code a message is
# trying to show. A fenced block wins over an inline span at the same position
# because the alternatives are tried left to right.
_PROTECTED_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?ms)"
    r"^(?P<fence>`{3,}|~{3,})[^\n]*\n.*?(?:^(?P=fence)[ \t]*$|\Z)"
    r"|`+[^`\n]*`+"
)


def _strip_mdx_syntax(content: str) -> str:
    """Remove MDX imports/exports and JSX tags, keeping literal code intact.

    Code fences and inline code spans are swapped for placeholders before the
    MDX-specific cleanup runs, then restored, so their contents are never
    rewritten into markdown or blanked out.
    """
    protected: list[str] = []

    def _protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"\x00{len(protected) - 1}\x00"

    masked = _PROTECTED_PATTERN.sub(_protect, content)
    masked = _IMPORT_PATTERN.sub("", masked)
    masked = _EXPORT_PATTERN.sub("", masked)
    masked = _JSX_TAG_PATTERN.sub("", masked)
    masked = _EXTRA_NEWLINES_PATTERN.sub("\n\n", masked)
    return re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], masked)


class MDXLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs: Any) -> LoaderResult:  # type: ignore[override]
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
        cleaned_content = _strip_mdx_syntax(content).strip()

        metadata = {"format": "mdx"}
        return LoaderResult(
            content=cleaned_content,
            source=source_ref,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=cleaned_content),
        )
