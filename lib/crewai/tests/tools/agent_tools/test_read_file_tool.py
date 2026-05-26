"""Unit tests for ReadFileTool."""

from __future__ import annotations

import base64
import io
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    NameObject,
)

from crewai.tools.agent_tools.read_file_tool import ReadFileTool
from crewai_files import ImageFile, PDFFile, TextFile


def _make_pdf(*page_texts: str) -> bytes:
    """Build a minimal valid PDF with extractable text on each page."""
    writer = PdfWriter()
    for text in page_texts:
        writer.add_blank_page(width=200, height=200)
        page = writer.pages[-1]

        font_dict = DictionaryObject()
        font_dict[NameObject("/Type")] = NameObject("/Font")
        font_dict[NameObject("/Subtype")] = NameObject("/Type1")
        font_dict[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font_ref = writer._add_object(font_dict)

        resources = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_ref
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources

        stream = DecodedStreamObject()
        escaped = text.replace("(", "\\(").replace(")", "\\)")
        stream.set_data(f"BT /F1 12 Tf 50 100 Td ({escaped}) Tj ET".encode())
        stream_ref = writer._add_object(stream)
        page[NameObject("/Contents")] = stream_ref

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestReadFileTool:
    """Tests for ReadFileTool."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.tool = ReadFileTool()

    def test_tool_metadata(self) -> None:
        """Test tool has correct name and description."""
        assert self.tool.name == "read_file"
        assert "Read content from an input file" in self.tool.description

    def test_run_no_files_available(self) -> None:
        """Test _run returns message when no files are set."""
        result = self.tool._run(file_name="any.txt")
        assert result == "No input files available."

    def test_run_file_not_found(self) -> None:
        """Test _run returns message when file not found."""
        self.tool.set_files({"doc.txt": TextFile(source=b"content")})

        result = self.tool._run(file_name="missing.txt")

        assert "File 'missing.txt' not found" in result
        assert "doc.txt" in result  # Lists available files

    def test_run_text_file(self) -> None:
        """Test reading a text file returns decoded content."""
        text_content = "Hello, this is text content!"
        self.tool.set_files({"readme.txt": TextFile(source=text_content.encode())})

        result = self.tool._run(file_name="readme.txt")

        assert result == text_content

    def test_run_json_file(self) -> None:
        """Test reading a JSON file returns decoded content."""
        json_content = '{"key": "value"}'
        self.tool.set_files({"data.json": TextFile(source=json_content.encode())})

        result = self.tool._run(file_name="data.json")

        assert result == json_content

    def test_run_binary_file_returns_base64(self) -> None:
        """Test reading a binary file returns base64 encoded content."""
        # Minimal valid PNG structure for proper MIME detection
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR"
            b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        self.tool.set_files({"image.png": ImageFile(source=png_bytes)})

        result = self.tool._run(file_name="image.png")

        assert "[Binary file:" in result
        assert "image/png" in result
        assert "Base64:" in result

        # Verify base64 can be decoded
        b64_part = result.split("Base64: ")[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == png_bytes

    def test_run_pdf_file_extracts_text(self) -> None:
        """Test reading a PDF extracts text instead of returning base64."""
        pdf_bytes = _make_pdf("Hello World from PDF")
        self.tool.set_files({"doc.pdf": PDFFile(source=pdf_bytes)})

        result = self.tool._run(file_name="doc.pdf")

        assert "Hello World from PDF" in result
        assert "Base64" not in result
        assert "--- Page 1 ---" in result

    def test_run_pdf_multipage_extracts_all_pages(self) -> None:
        """Test reading a multi-page PDF extracts text from every page."""
        pdf_bytes = _make_pdf("First page content", "Second page content")
        self.tool.set_files({"report.pdf": PDFFile(source=pdf_bytes)})

        result = self.tool._run(file_name="report.pdf")

        assert "First page content" in result
        assert "Second page content" in result
        assert "--- Page 1 ---" in result
        assert "--- Page 2 ---" in result
        assert "Base64" not in result

    def test_run_pdf_no_extractable_text(self) -> None:
        """Test PDF with no extractable text returns a friendly message."""
        # A blank page with no text content
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        buf = io.BytesIO()
        writer.write(buf)
        blank_pdf = buf.getvalue()

        self.tool.set_files({"blank.pdf": PDFFile(source=blank_pdf)})

        result = self.tool._run(file_name="blank.pdf")

        assert "No extractable text found" in result
        assert "Base64" not in result

    def test_run_pdf_corrupted_returns_error_message(self) -> None:
        """Test that a corrupted PDF returns a short error, never base64."""
        corrupted = b"%PDF-1.4 this is not a valid PDF structure"
        self.tool.set_files({"bad.pdf": PDFFile(source=corrupted)})

        result = self.tool._run(file_name="bad.pdf")

        assert "[PDF file: bad.pdf]" in result
        assert "Failed to extract text" in result
        assert "Base64" not in result

    def test_run_pdf_no_pypdf_returns_install_message(self) -> None:
        """Test graceful fallback when pypdf is not installed."""
        pdf_bytes = _make_pdf("Some text")
        self.tool.set_files({"doc.pdf": PDFFile(source=pdf_bytes)})

        with patch.dict("sys.modules", {"pypdf": None}):
            result = self.tool._run(file_name="doc.pdf")

        assert "pypdf is not installed" in result
        assert "Base64" not in result

    def test_run_pdf_result_much_smaller_than_base64(self) -> None:
        """Extracted text should be far smaller than a base64-encoded PDF."""
        pdf_bytes = _make_pdf("Short text")
        self.tool.set_files({"doc.pdf": PDFFile(source=pdf_bytes)})

        result = self.tool._run(file_name="doc.pdf")

        base64_size = len(base64.b64encode(pdf_bytes))
        assert len(result) < base64_size

    def test_set_files_none(self) -> None:
        """Test setting files to None."""
        self.tool.set_files({"doc": TextFile(source=b"content")})
        self.tool.set_files(None)

        result = self.tool._run(file_name="doc")

        assert result == "No input files available."

    def test_run_multiple_files(self) -> None:
        """Test tool can access multiple files."""
        self.tool.set_files({
            "file1.txt": TextFile(source=b"content 1"),
            "file2.txt": TextFile(source=b"content 2"),
            "file3.txt": TextFile(source=b"content 3"),
        })

        assert self.tool._run(file_name="file1.txt") == "content 1"
        assert self.tool._run(file_name="file2.txt") == "content 2"
        assert self.tool._run(file_name="file3.txt") == "content 3"

    def test_run_with_kwargs(self) -> None:
        """Test _run ignores extra kwargs."""
        self.tool.set_files({"doc.txt": TextFile(source=b"content")})

        result = self.tool._run(file_name="doc.txt", extra_arg="ignored")

        assert result == "content"

    def test_args_schema(self) -> None:
        """Test that args_schema is properly defined."""
        schema = self.tool.args_schema

        assert "file_name" in schema.model_fields
        assert schema.model_fields["file_name"].is_required()
