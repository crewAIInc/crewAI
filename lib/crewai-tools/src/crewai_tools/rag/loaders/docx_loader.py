import os
import tempfile
from typing import Any

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent
from crewai_tools.security.safe_requests import safe_get


class DOCXLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs: Any) -> LoaderResult:  # type: ignore[override]
        try:
            from docx import Document as DocxDocument
        except ImportError as e:
            raise ImportError(
                "python-docx is required for DOCX loading. Install with: 'uv pip install python-docx' or pip install crewai-tools[rag]"
            ) from e

        source_ref = source_content.source_ref

        if source_content.is_url():
            temp_file = self._download_from_url(source_ref, kwargs)
            try:
                return self._load_from_file(temp_file, source_ref, DocxDocument)
            finally:
                os.unlink(temp_file)
        elif source_content.path_exists():
            return self._load_from_file(source_ref, source_ref, DocxDocument)
        else:
            raise ValueError(
                f"Source must be a valid file path or URL, got: {source_content.source}"
            )

    @staticmethod
    def _download_from_url(url: str, kwargs: dict[str, Any]) -> str:
        headers = kwargs.get(
            "headers",
            {
                "Accept": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "User-Agent": "Mozilla/5.0 (compatible; crewai-tools DOCXLoader)",
            },
        )

        try:
            response = safe_get(url, headers=headers, timeout=30)
            response.raise_for_status()

            # Create temporary file to save the DOCX content
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_file:
                temp_file.write(response.content)
                return temp_file.name
        except Exception as e:
            raise ValueError(f"Error fetching content from URL {url}: {e!s}") from e

    def _load_from_file(
        self,
        file_path: str,
        source_ref: str,
        DocxDocument: Any,  # noqa: N803
    ) -> LoaderResult:
        try:
            doc = DocxDocument(file_path)

            # Walk paragraphs and tables in document order so table text is
            # kept next to the text that introduces it.
            text_parts = [
                line
                for block in doc.iter_inner_content()
                for line in self._block_lines(block)
            ]

            content = "\n".join(text_parts)

            metadata = {
                "format": "docx",
                "paragraphs": len(doc.paragraphs),
                "tables": len(doc.tables),
            }

            return LoaderResult(
                content=content,
                source=source_ref,
                metadata=metadata,
                doc_id=self.generate_doc_id(source_ref=source_ref, content=content),
            )

        except Exception as e:
            raise ValueError(f"Error loading DOCX file: {e!s}") from e

    @classmethod
    def _block_lines(cls, block: Any) -> list[str]:
        """Return the non-empty text lines of a paragraph or table.

        A table yields one line per row with cells joined by `` | ``. Each
        physical cell is read once: a cell spanning several columns appears
        once, and the continuation of a vertically merged cell is left empty.
        Line breaks inside a cell and tables nested in it are flattened into
        the cell's text, so a row always stays on one line.
        """
        from docx.table import Table, _Cell

        if not isinstance(block, Table):
            text = block.text
            return [text] if text.strip() else []

        lines = []
        for row in block.rows:
            # Iterate the row's own <w:tc> elements rather than ``row.cells``,
            # which repeats a cell once per spanned grid column and maps a
            # vertical-merge continuation back to the merged cell, re-reading
            # its content (and any nested tables) for every row it spans.
            row_text = []
            for tc in row._tr.tc_lst:
                if tc.vMerge == "continue":
                    row_text.append("")
                    continue
                cell_lines = [
                    line
                    for inner in _Cell(tc, block).iter_inner_content()
                    for line in cls._block_lines(inner)
                ]
                row_text.append(" ".join(" ".join(cell_lines).split()))
            if any(row_text):
                lines.append(" | ".join(row_text))
        return lines
