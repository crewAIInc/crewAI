"""Tests for experimental environment tools."""

import os
import tempfile
from pathlib import Path

import pytest

from crewai.experimental.environment_tools import (
    EnvironmentTools,
    FileReadTool,
    FileSearchTool,
    GrepTool,
    ListDirTool,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")

        python_file = Path(tmpdir) / "example.py"
        python_file.write_text("def hello():\n    print('Hello World')\n")

        # Create subdirectory with files
        subdir = Path(tmpdir) / "subdir"
        subdir.mkdir()
        (subdir / "nested.txt").write_text("Nested content\n")
        (subdir / "another.py").write_text("# Another Python file\n")

        yield tmpdir


@pytest.fixture
def restricted_temp_dir():
    """Create two directories - one allowed, one not."""
    with tempfile.TemporaryDirectory() as allowed_dir:
        with tempfile.TemporaryDirectory() as forbidden_dir:
            # Create files in both
            (Path(allowed_dir) / "allowed.txt").write_text("Allowed content\n")
            (Path(forbidden_dir) / "forbidden.txt").write_text("Forbidden content\n")

            yield allowed_dir, forbidden_dir


# ============================================================================
# BaseEnvironmentTool Tests
# ============================================================================


class TestBaseEnvironmentTool:
    """Tests for BaseEnvironmentTool path validation."""

    def test_validate_path_no_restrictions(self, temp_dir):
        """With no allowed_paths, all paths should be allowed."""
        tool = FileReadTool(allowed_paths=[])
        valid, result = tool._validate_path(temp_dir)

        assert valid is True
        assert isinstance(result, Path)

    def test_validate_path_within_allowed(self, temp_dir):
        """Paths within allowed_paths should be valid."""
        tool = FileReadTool(allowed_paths=[temp_dir])
        test_file = os.path.join(temp_dir, "test.txt")

        valid, result = tool._validate_path(test_file)

        assert valid is True
        assert isinstance(result, Path)

    def test_validate_path_outside_allowed(self, restricted_temp_dir):
        """Paths outside allowed_paths should be rejected."""
        allowed_dir, forbidden_dir = restricted_temp_dir
        tool = FileReadTool(allowed_paths=[allowed_dir])

        forbidden_file = os.path.join(forbidden_dir, "forbidden.txt")
        valid, result = tool._validate_path(forbidden_file)

        assert valid is False
        assert "outside allowed paths" in result

    def test_format_size(self):
        """Test human-readable size formatting."""
        tool = FileReadTool()

        assert tool._format_size(500) == "500B"
        assert tool._format_size(1024) == "1.0KB"
        assert tool._format_size(1536) == "1.5KB"
        assert tool._format_size(1024 * 1024) == "1.0MB"
        assert tool._format_size(1024 * 1024 * 1024) == "1.0GB"


# ============================================================================
# FileReadTool Tests
# ============================================================================


class TestFileReadTool:
    """Tests for FileReadTool."""

    def test_read_entire_file(self, temp_dir):
        """Should read entire file contents."""
        tool = FileReadTool()
        test_file = os.path.join(temp_dir, "test.txt")

        result = tool._run(path=test_file)

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 5" in result
        assert "File:" in result  # Metadata header

    def test_read_with_line_range(self, temp_dir):
        """Should read specific line range."""
        tool = FileReadTool()
        test_file = os.path.join(temp_dir, "test.txt")

        result = tool._run(path=test_file, start_line=2, line_count=2)

        assert "Line 2" in result
        assert "Line 3" in result
        # Should not include lines outside range
        assert "Line 1" not in result.split("=" * 60)[-1]  # Check content after header

    def test_read_file_not_found(self, temp_dir):
        """Should return error for missing file."""
        tool = FileReadTool()
        missing_file = os.path.join(temp_dir, "nonexistent.txt")

        result = tool._run(path=missing_file)

        assert "Error: File not found" in result

    def test_read_file_path_restricted(self, restricted_temp_dir):
        """Should reject paths outside allowed_paths."""
        allowed_dir, forbidden_dir = restricted_temp_dir
        tool = FileReadTool(allowed_paths=[allowed_dir])

        forbidden_file = os.path.join(forbidden_dir, "forbidden.txt")
        result = tool._run(path=forbidden_file)

        assert "Error:" in result
        assert "outside allowed paths" in result


# ============================================================================
# ListDirTool Tests
# ============================================================================


class TestListDirTool:
    """Tests for ListDirTool."""

    def test_list_directory(self, temp_dir):
        """Should list directory contents."""
        tool = ListDirTool()

        result = tool._run(path=temp_dir)

        assert "test.txt" in result
        assert "example.py" in result
        assert "subdir" in result
        assert "Total:" in result

    def test_list_with_pattern(self, temp_dir):
        """Should filter by pattern."""
        tool = ListDirTool()

        result = tool._run(path=temp_dir, pattern="*.py")

        assert "example.py" in result
        assert "test.txt" not in result

    def test_list_recursive(self, temp_dir):
        """Should list recursively when enabled."""
        tool = ListDirTool()

        result = tool._run(path=temp_dir, recursive=True)

        assert "nested.txt" in result
        assert "another.py" in result

    def test_list_nonexistent_directory(self, temp_dir):
        """Should return error for missing directory."""
        tool = ListDirTool()

        result = tool._run(path=os.path.join(temp_dir, "nonexistent"))

        assert "Error: Directory not found" in result

    def test_list_path_restricted(self, restricted_temp_dir):
        """Should reject paths outside allowed_paths."""
        allowed_dir, forbidden_dir = restricted_temp_dir
        tool = ListDirTool(allowed_paths=[allowed_dir])

        result = tool._run(path=forbidden_dir)

        assert "Error:" in result
        assert "outside allowed paths" in result


# ============================================================================
# GrepTool Tests
# ============================================================================


class TestGrepTool:
    """Tests for GrepTool."""

    def test_grep_finds_pattern(self, temp_dir):
        """Should find matching patterns."""
        tool = GrepTool()
        test_file = os.path.join(temp_dir, "test.txt")

        result = tool._run(pattern="Line 2", path=test_file)

        assert "Line 2" in result
        assert "matches" in result.lower() or "found" in result.lower()

    def test_grep_no_matches(self, temp_dir):
        """Should report when no matches found."""
        tool = GrepTool()
        test_file = os.path.join(temp_dir, "test.txt")

        result = tool._run(pattern="nonexistent pattern xyz", path=test_file)

        assert "No matches found" in result

    def test_grep_recursive(self, temp_dir):
        """Should search recursively in directories."""
        tool = GrepTool()

        result = tool._run(pattern="Nested", path=temp_dir, recursive=True)

        assert "Nested" in result

    def test_grep_case_insensitive(self, temp_dir):
        """Should support case-insensitive search."""
        tool = GrepTool()
        test_file = os.path.join(temp_dir, "test.txt")

        result = tool._run(pattern="LINE", path=test_file, ignore_case=True)

        assert "Line" in result or "matches" in result.lower()

    def test_grep_path_restricted(self, restricted_temp_dir):
        """Should reject paths outside allowed_paths."""
        allowed_dir, forbidden_dir = restricted_temp_dir
        tool = GrepTool(allowed_paths=[allowed_dir])

        result = tool._run(pattern="test", path=forbidden_dir)

        assert "Error:" in result
        assert "outside allowed paths" in result


# ============================================================================
# FileSearchTool Tests
# ============================================================================


class TestFileSearchTool:
    """Tests for FileSearchTool."""

    def test_find_files_by_pattern(self, temp_dir):
        """Should find files matching pattern."""
        tool = FileSearchTool()

        result = tool._run(pattern="*.py", path=temp_dir)

        assert "example.py" in result
        assert "another.py" in result

    def test_find_no_matches(self, temp_dir):
        """Should report when no files match."""
        tool = FileSearchTool()

        result = tool._run(pattern="*.xyz", path=temp_dir)

        assert "No" in result and "found" in result

    def test_find_files_only(self, temp_dir):
        """Should filter to files only."""
        tool = FileSearchTool()

        result = tool._run(pattern="*", path=temp_dir, file_type="file")

        # Should include files
        assert "test.txt" in result or "example.py" in result
        # Directories should have trailing slash in output
        # Check that subdir is not listed as a file

    def test_find_dirs_only(self, temp_dir):
        """Should filter to directories only."""
        tool = FileSearchTool()

        result = tool._run(pattern="*", path=temp_dir, file_type="dir")

        assert "subdir" in result

    def test_find_path_restricted(self, restricted_temp_dir):
        """Should reject paths outside allowed_paths."""
        allowed_dir, forbidden_dir = restricted_temp_dir
        tool = FileSearchTool(allowed_paths=[allowed_dir])

        result = tool._run(pattern="*", path=forbidden_dir)

        assert "Error:" in result
        assert "outside allowed paths" in result


# ============================================================================
# EnvironmentTools Manager Tests
# ============================================================================


class TestEnvironmentTools:
    """Tests for EnvironmentTools manager class."""

    def test_returns_all_tools_by_default(self):
        """Should return all four tools by default."""
        env_tools = EnvironmentTools()
        tools = env_tools.tools()

        assert len(tools) == 4

        tool_names = [t.name for t in tools]
        assert "read_file" in tool_names
        assert "list_directory" in tool_names
        assert "grep_search" in tool_names
        assert "find_files" in tool_names

    def test_exclude_grep(self):
        """Should exclude grep tool when disabled."""
        env_tools = EnvironmentTools(include_grep=False)
        tools = env_tools.tools()

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "grep_search" not in tool_names

    def test_exclude_search(self):
        """Should exclude search tool when disabled."""
        env_tools = EnvironmentTools(include_search=False)
        tools = env_tools.tools()

        assert len(tools) == 3
        tool_names = [t.name for t in tools]
        assert "find_files" not in tool_names

    def test_allowed_paths_propagated(self, temp_dir):
        """Should propagate allowed_paths to all tools."""
        env_tools = EnvironmentTools(allowed_paths=[temp_dir])
        tools = env_tools.tools()

        for tool in tools:
            assert tool.allowed_paths == [temp_dir]

    def test_tools_are_base_tool_instances(self):
        """All returned tools should be BaseTool instances."""
        from crewai.tools.base_tool import BaseTool

        env_tools = EnvironmentTools()
        tools = env_tools.tools()

        for tool in tools:
            assert isinstance(tool, BaseTool)
