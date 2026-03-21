"""Unit tests for FileReadTool."""

from pathlib import Path

import pytest

from crewai_tools.tools.file_read_tool.file_read_tool import FileReadTool


@pytest.fixture
def sample_file(tmp_path: Path) -> Path:
    """Create a sample text file with numbered lines."""
    file_path = tmp_path / "sample.txt"
    lines = [f"Line {i}: This is line number {i}." for i in range(1, 101)]
    file_path.write_text("\n".join(lines) + "\n")
    return file_path


@pytest.fixture
def binary_file(tmp_path: Path) -> Path:
    """Create a binary file with null bytes."""
    file_path = tmp_path / "binary.bin"
    file_path.write_bytes(b"\x00\x01\x02\x03binary content\x00\x04\x05")
    return file_path


@pytest.fixture
def empty_file(tmp_path: Path) -> Path:
    """Create an empty file."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    return file_path


class TestFileReadTool:
    """Tests for FileReadTool."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.tool = FileReadTool()

    def test_tool_metadata(self) -> None:
        """Test tool has correct name and description."""
        assert self.tool.name == "read_file"
        assert "read" in self.tool.description.lower()

    def test_args_schema(self) -> None:
        """Test that args_schema has correct fields."""
        schema = self.tool.args_schema
        fields = schema.model_fields

        assert "file_path" in fields
        assert fields["file_path"].is_required()

        assert "offset" in fields
        assert not fields["offset"].is_required()

        assert "limit" in fields
        assert not fields["limit"].is_required()

        assert "include_line_numbers" in fields
        assert not fields["include_line_numbers"].is_required()

    def test_read_entire_file(self, sample_file: Path) -> None:
        """Test reading entire file with line numbers."""
        result = self.tool._run(file_path=str(sample_file))
        assert "File:" in result
        assert "Total lines: 100" in result
        assert "Line 1:" in result
        assert "|" in result  # Line number separator

    def test_read_with_offset(self, sample_file: Path) -> None:
        """Test reading from a specific line offset."""
        result = self.tool._run(file_path=str(sample_file), offset=50, limit=10)
        assert "Showing lines: 50-59" in result
        assert "Line 50:" in result
        assert "Line 59:" in result
        # Should not include lines before offset
        assert "Line 49:" not in result

    def test_negative_offset_reads_from_end(self, sample_file: Path) -> None:
        """Test negative offset reads from end of file."""
        result = self.tool._run(file_path=str(sample_file), offset=-10)
        assert "Showing lines: 91-100" in result
        assert "Line 91:" in result
        assert "Line 100:" in result

    def test_limit_controls_line_count(self, sample_file: Path) -> None:
        """Test limit parameter controls how many lines are read."""
        result = self.tool._run(file_path=str(sample_file), offset=1, limit=5)
        assert "Showing lines: 1-5" in result
        # Count output lines (excluding header)
        content_lines = [l for l in result.split("\n") if "|" in l and l.strip()]
        assert len(content_lines) == 5

    def test_line_numbers_included_by_default(self, sample_file: Path) -> None:
        """Test line numbers are included by default."""
        result = self.tool._run(file_path=str(sample_file), limit=5)
        # Lines should have format "     1|content"
        assert "|" in result
        for line in result.split("\n"):
            if "Line 1:" in line:
                assert "|" in line

    def test_line_numbers_can_be_disabled(self, sample_file: Path) -> None:
        """Test line numbers can be disabled."""
        result = self.tool._run(
            file_path=str(sample_file), limit=5, include_line_numbers=False
        )
        # Content lines shouldn't have the line number prefix
        content_section = result.split("\n\n", 1)[-1]  # Skip header
        for line in content_section.split("\n"):
            if line.strip() and "Line" in line:
                # Should not start with number|
                assert not line.strip()[0].isdigit() or "|" not in line[:10]

    def test_binary_file_detection(self, binary_file: Path) -> None:
        """Test binary files are detected and not read as text."""
        result = self.tool._run(file_path=str(binary_file))
        assert "Error" in result
        assert "binary" in result.lower()

    def test_empty_file(self, empty_file: Path) -> None:
        """Test reading empty file returns appropriate message."""
        result = self.tool._run(file_path=str(empty_file))
        assert "Total lines: 0" in result
        assert "Empty file" in result

    def test_file_not_found(self) -> None:
        """Test error message when file doesn't exist."""
        result = self.tool._run(file_path="/nonexistent/file.txt")
        assert "Error" in result
        assert "not found" in result.lower()

    def test_directory_path_error(self, tmp_path: Path) -> None:
        """Test error when path is a directory."""
        result = self.tool._run(file_path=str(tmp_path))
        assert "Error" in result
        assert "directory" in result.lower()

    def test_file_metadata_in_header(self, sample_file: Path) -> None:
        """Test file metadata is included in response header."""
        result = self.tool._run(file_path=str(sample_file), limit=10)
        # Should have file path
        assert str(sample_file) in result
        # Should have total lines
        assert "Total lines:" in result

    def test_large_file_auto_truncation(self, tmp_path: Path) -> None:
        """Test large files are automatically truncated."""
        # Create a file with 1000 lines
        large_file = tmp_path / "large.txt"
        lines = [f"Line {i}" for i in range(1, 1001)]
        large_file.write_text("\n".join(lines))

        result = self.tool._run(file_path=str(large_file))
        # Should be truncated and include message about it
        assert "truncated" in result.lower() or "Showing lines" in result
        # Should not read all 1000 lines without explicit limit
        assert "Line 1000" not in result or "limit" in result.lower()

    def test_legacy_start_line_parameter(self, sample_file: Path) -> None:
        """Test backward compatibility with start_line parameter."""
        result = self.tool._run(file_path=str(sample_file), start_line=10, line_count=5)
        assert "Showing lines: 10-14" in result
        assert "Line 10:" in result

    def test_constructor_with_file_path(self, sample_file: Path) -> None:
        """Test constructing tool with default file path."""
        tool = FileReadTool(file_path=str(sample_file))
        result = tool._run()
        assert "Line 1:" in result

    def test_constructor_file_path_override(self, sample_file: Path, tmp_path: Path) -> None:
        """Test runtime file_path overrides constructor file_path."""
        other_file = tmp_path / "other.txt"
        other_file.write_text("Different content\n")

        tool = FileReadTool(file_path=str(sample_file))
        result = tool._run(file_path=str(other_file))
        assert "Different content" in result
        assert "Line 1:" not in result

    def test_no_file_path_error(self) -> None:
        """Test error when no file path is provided."""
        result = self.tool._run()
        assert "Error" in result
        assert "No file path" in result

    def test_offset_beyond_file_length(self, sample_file: Path) -> None:
        """Test offset beyond file length returns empty content."""
        result = self.tool._run(file_path=str(sample_file), offset=200)
        # File has 100 lines, offset 200 should show nothing
        # But header should still show file info
        assert "Total lines: 100" in result
