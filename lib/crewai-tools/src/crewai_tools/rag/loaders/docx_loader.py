from __future__ import annotations

from collections.abc import Iterator
import os
import tempfile
from typing import TYPE_CHECKING, Any

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent
from crewai_tools.security.safe_requests import safe_get


if TYPE_CHECKING:
    from collections.abc import Iterable

    from docx.oxml.table import CT_Tc
    from docx.table import Table
    from docx.text.paragraph import Paragraph


class DOCXLoader(BaseLoader):
    def load(self, source_content: SourceContent, **kwargs: Any) -> LoaderResult:  # type: ignore[override]
        """Load DOCX text from a local file or URL and clean up downloaded files."""
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
        """Download DOCX content to a temporary file after checking HTTP status."""
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
        """Extract ordered paragraph and table text while retaining source metadata."""
        try:
            doc = DocxDocument(file_path)

            content = "\n".join(self._iter_text(doc.iter_inner_content()))

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

    def _iter_text(self, blocks: Iterable[Paragraph | Table]) -> Iterator[str]:
        """Yield paragraph and table text in document order, including nested tables."""
        from docx.text.paragraph import Paragraph

        for block in blocks:
            if isinstance(block, Paragraph):
                if block.text.strip():
                    yield block.text
            else:
                # Merged grid positions can refer to the same cell across rows.
                seen_cells: set[CT_Tc] = set()
                for row in block.rows:
                    cells = []
                    for cell in row.cells:
                        if cell._tc in seen_cells:
                            continue
                        seen_cells.add(cell._tc)
                        cells.append(
                            "\n".join(self._iter_text(cell.iter_inner_content()))
                        )
                    if any(cell.strip() for cell in cells):
                        yield " | ".join(cells)
