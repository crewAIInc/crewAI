"""Tests for DataTypes.from_content extension auto-detection."""

from pathlib import Path

import pytest

from crewai_tools.rag.data_types import DataType, DataTypes


class TestFromContentCaseInsensitiveExtensions:
    """Regression tests for #6399: uppercase/mixed-case extensions."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("report.pdf", DataType.PDF_FILE),
            ("report.PDF", DataType.PDF_FILE),
            ("report.Pdf", DataType.PDF_FILE),
            ("data.csv", DataType.CSV),
            ("data.CSV", DataType.CSV),
            ("doc.docx", DataType.DOCX),
            ("doc.DOCX", DataType.DOCX),
            ("notes.md", DataType.MDX),
            ("notes.MD", DataType.MDX),
            ("notes.mdx", DataType.MDX),
            ("notes.MDX", DataType.MDX),
            ("payload.json", DataType.JSON),
            ("payload.JSON", DataType.JSON),
            ("tree.xml", DataType.XML),
            ("tree.XML", DataType.XML),
            ("readme.txt", DataType.TEXT_FILE),
            ("readme.TXT", DataType.TEXT_FILE),
        ],
    )
    def test_extension_detection_is_case_insensitive(
        self, path: str, expected: DataType, tmp_path: Path
    ) -> None:
        """Local file paths with mixed-case extensions map to the right DataType."""
        file_path = tmp_path / path
        file_path.write_bytes(b"dummy")
        assert DataTypes.from_content(str(file_path)) == expected

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            ("https://example.com/file.pdf", DataType.PDF_FILE),
            ("https://example.com/file.PDF", DataType.PDF_FILE),
            ("https://example.com/data.CSV", DataType.CSV),
            ("https://example.com/doc.DOCX", DataType.DOCX),
            ("https://example.com/payload.JSON", DataType.JSON),
        ],
    )
    def test_url_extension_detection_is_case_insensitive(
        self, url: str, expected: DataType
    ) -> None:
        """URL path extensions must also be matched case-insensitively."""
        assert DataTypes.from_content(url) == expected

    def test_url_without_known_extension_stays_website(self) -> None:
        """URLs without a known file extension still classify as website."""
        assert (
            DataTypes.from_content("https://example.com/docs/page") == DataType.WEBSITE
        )

    def test_path_object_with_uppercase_extension(self, tmp_path: Path) -> None:
        """Path objects with uppercase suffixes are handled correctly."""
        file_path = tmp_path / "Report.PDF"
        file_path.write_bytes(b"%PDF")
        assert DataTypes.from_content(file_path) == DataType.PDF_FILE
