import re
from collections.abc import Callable, Iterator
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

# A fence line is up to three spaces of indentation followed by a run of at
# least three backticks or tildes. The closing fence must reuse the same
# character and be at least as long as the opening one.
_FENCE_LINE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<indent> {0,3})(?P<fence>`{3,}|~{3,})(?P<info>[^\n]*)$"
)


def _iter_blocks(content: str) -> Iterator[tuple[bool, str]]:
    """Split ``content`` into ``(is_code, text)`` pairs.

    Fenced code blocks (backtick or tilde fences, any length) are yielded
    verbatim as code, everything else as prose. An unterminated fence is
    treated as code running to the end of the document, matching how
    Markdown renders it.
    """
    lines = content.splitlines(keepends=True)
    pending: list[str] = []
    index = 0

    while index < len(lines):
        opener = _FENCE_LINE_PATTERN.match(lines[index].rstrip("\r\n"))
        if opener is None:
            pending.append(lines[index])
            index += 1
            continue

        fence = opener.group("fence")
        marker = fence[0]

        # An info string cannot contain the fence character itself.
        if marker == "`" and marker in opener.group("info"):
            pending.append(lines[index])
            index += 1
            continue

        closer = re.compile(
            r"^ {0,3}" + re.escape(marker) + "{" + str(len(fence)) + r",}[ \t]*$"
        )
        end = index + 1
        while end < len(lines) and not closer.match(lines[end].rstrip("\r\n")):
            end += 1

        if pending:
            yield False, "".join(pending)
            pending = []

        if end >= len(lines):
            yield True, "".join(lines[index:])
            return

        yield True, "".join(lines[index : end + 1])
        index = end + 1

    if pending:
        yield False, "".join(pending)


def _iter_inline_code(
    text: str, is_inside_tag: Callable[[int], bool] | None = None
) -> Iterator[tuple[bool, str]]:
    """Split prose into ``(is_code, text)`` pairs.

    Inline code spans are delimited by a run of backticks closed by a run of
    the same length, so ``x`` and ``` `` ``a`b`` `` ``` are handled correctly.
    Backtick runs without a matching closer stay in the prose. When
    ``is_inside_tag`` is given, backticks for which it returns ``True`` are
    ignored, so an attribute value such as ``<Foo bar={`x`} />`` is not read as
    Markdown inline code.
    """
    start = 0
    index = 0
    length = len(text)

    def _is_ignored(position: int) -> bool:
        return is_inside_tag is not None and is_inside_tag(position)

    while index < length:
        if text[index] != "`" or _is_ignored(index):
            index += 1
            continue

        open_end = index
        while open_end < length and text[open_end] == "`":
            open_end += 1
        run = open_end - index

        scan = open_end
        close_start = -1
        while scan < length:
            if text[scan] != "`" or _is_ignored(scan):
                scan += 1
                continue
            run_end = scan
            while run_end < length and text[run_end] == "`":
                run_end += 1
            if run_end - scan == run:
                close_start = scan
                break
            scan = run_end

        if close_start == -1:
            index = open_end
            continue

        if index > start:
            yield False, text[start:index]
        yield True, text[index : close_start + run]
        index = close_start + run
        start = index

    if start < length:
        yield False, text[start:]


def _clean_prose(text: str) -> str:
    """Strip real MDX syntax from a prose segment, leaving code untouched."""
    # Backticks that sit inside a JSX/HTML tag are attribute values rather than
    # Markdown inline code, so collect the tag spans up front and ignore any
    # backtick that falls inside one.
    tag_ranges = [match.span() for match in _JSX_TAG_PATTERN.finditer(text)]

    def is_inside_tag(position: int) -> bool:
        return any(start <= position < end for start, end in tag_ranges)

    out: list[str] = []
    for is_code, segment in _iter_inline_code(text, is_inside_tag):
        if is_code:
            out.append(segment)
            continue
        cleaned = _IMPORT_PATTERN.sub("", segment)
        cleaned = _EXPORT_PATTERN.sub("", cleaned)
        cleaned = _JSX_TAG_PATTERN.sub("", cleaned)
        out.append(_EXTRA_NEWLINES_PATTERN.sub("\n\n", cleaned))
    return "".join(out)


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
        parts: list[str] = []
        for is_code, block in _iter_blocks(content):
            parts.append(block if is_code else _clean_prose(block))
        cleaned_content = "".join(parts).strip()

        metadata = {"format": "mdx"}
        return LoaderResult(
            content=cleaned_content,
            source=source_ref,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=source_ref, content=cleaned_content),
        )
