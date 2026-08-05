"""Tests for file-type detection in ``DataTypes.from_content``.

Extension detection must be case-insensitive so that files/URLs with
uppercase or mixed-case extensions (e.g. ``report.PDF``) are routed to the
correct loader instead of silently falling back to the plain-text loader.
"""

from pathlib import Path
import tempfile

import pytest

from crewai_tools.rag.data_types import DataType, DataTypes


class TestFileExtensionCaseInsensitivity:
    """Local-file extension detection should ignore case."""

    @pytest.mark.parametrize(
        ("suffix", "expected"),
        [
            (".pdf", DataType.PDF_FILE),
            (".PDF", DataType.PDF_FILE),
            (".Pdf", DataType.PDF_FILE),
            (".csv", DataType.CSV),
            (".CSV", DataType.CSV),
            (".json", DataType.JSON),
            (".JSON", DataType.JSON),
            (".xml", DataType.XML),
            (".XML", DataType.XML),
            (".docx", DataType.DOCX),
            (".DOCX", DataType.DOCX),
            (".mdx", DataType.MDX),
            (".MDX", DataType.MDX),
            (".md", DataType.MDX),
            (".MD", DataType.MDX),
            (".txt", DataType.TEXT_FILE),
            (".TXT", DataType.TEXT_FILE),
        ],
    )
    def test_local_file_extension_is_case_insensitive(
        self, suffix: str, expected: DataType
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / f"document{suffix}"
            file_path.write_text("content")

            assert DataTypes.from_content(str(file_path)) == expected

    def test_uppercase_pdf_is_not_misdetected_as_text_file(self) -> None:
        """Regression: an uppercase ``.PDF`` must not fall back to text_file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "report.PDF"
            file_path.write_text("content")

            assert DataTypes.from_content(str(file_path)) == DataType.PDF_FILE


class TestUrlExtensionCaseInsensitivity:
    """URL extension detection should ignore case."""

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://example.com/file.pdf", DataType.PDF_FILE),
            ("https://example.com/file.PDF", DataType.PDF_FILE),
            ("https://example.com/data.CSV", DataType.CSV),
            ("https://example.com/data.JSON", DataType.JSON),
        ],
    )
    def test_url_extension_is_case_insensitive(
        self, url: str, expected: DataType
    ) -> None:
        assert DataTypes.from_content(url) == expected
