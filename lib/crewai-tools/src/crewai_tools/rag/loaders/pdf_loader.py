"""PDF loader for extracting text from PDF files."""

import os
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse
import urllib.request

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class PDFLoader(BaseLoader):
    """Loader for PDF files and URLs."""

    @staticmethod
    def _is_url(path: str) -> bool:
        """Check if the path is a URL."""
        try:
            parsed = urlparse(path)
            return parsed.scheme in ("http", "https")
        except Exception:
            return False

    @staticmethod
    def _download_pdf(url: str) -> bytes:
        """Download PDF content from a URL.

        Args:
            url: The URL to download from.

        Returns:
            The PDF content as bytes.

        Raises:
            ValueError: If the download fails.
        """

        try:
            with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
                return cast(bytes, response.read())
        except Exception as e:
            raise ValueError(f"Failed to download PDF from {url}: {e!s}") from e

    def load(self, source: SourceContent, **kwargs: Any) -> LoaderResult:  # type: ignore[override]
        """Load and extract text from a PDF file or URL.

        Args:
            source: The source content containing the PDF file path or URL.

        Returns:
            LoaderResult with extracted text content.

        Raises:
            FileNotFoundError: If the PDF file doesn't exist.
            ImportError: If required PDF libraries aren't installed.
            ValueError: If the PDF cannot be read or downloaded.
        """
        try:
            import pymupdf  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
                "PDF support requires pymupdf. Install with: uv add pymupdf"
            ) from e

        file_path = source.source
        is_url = self._is_url(file_path)

        if is_url:
            source_name = Path(urlparse(file_path).path).name or "downloaded.pdf"
        else:
            source_name = Path(file_path).name

        text_content: list[str] = []
        metadata: dict[str, Any] = {
            "source": file_path,
            "file_name": source_name,
            "file_type": "pdf",
        }

        try:
            if is_url:
                pdf_bytes = self._download_pdf(file_path)
                doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
            else:
                if not os.path.isfile(file_path):
                    raise FileNotFoundError(f"PDF file not found: {file_path}")
                doc = pymupdf.open(file_path)

            metadata["num_pages"] = len(doc)

            for page_num, page in enumerate(doc, 1):
                page_text = page.get_text()
                if page_text.strip():
                    text_content.append(f"Page {page_num}:\n{page_text}")

            doc.close()
        except FileNotFoundError:
            raise
        except Exception as e:
            raise ValueError(f"Error reading PDF from {file_path}: {e!s}") from e

        if not text_content:
            content = f"[PDF file with no extractable text: {source_name}]"
        else:
            content = "\n\n".join(text_content)

        return LoaderResult(
            content=content,
            source=file_path,
            metadata=metadata,
            doc_id=self.generate_doc_id(source_ref=file_path, content=content),
        )
