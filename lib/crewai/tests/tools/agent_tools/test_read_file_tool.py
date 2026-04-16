"""Unit tests for ReadFileTool."""

import base64

import pytest

from crewai.tools.agent_tools.read_file_tool import ReadFileTool
from crewai_files import ImageFile, PDFFile, TextFile


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

    def test_run_pdf_file_returns_base64(self) -> None:
        """Test reading a PDF file returns base64 encoded content."""
        pdf_bytes = b"%PDF-1.4 some content here"
        self.tool.set_files({"doc.pdf": PDFFile(source=pdf_bytes)})

        result = self.tool._run(file_name="doc.pdf")

        assert "[Binary file:" in result
        assert "application/pdf" in result

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