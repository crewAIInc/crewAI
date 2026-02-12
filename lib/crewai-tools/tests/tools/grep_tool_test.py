"""Unit tests for GrepTool."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from crewai_tools import GrepTool
from crewai_tools.tools.grep_tool.grep_tool import (
    MAX_CONTEXT_LINES,
    MAX_REGEX_LENGTH,
    GrepToolSchema,
)


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample files for testing."""
    # src/main.py
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text(
        "def hello():\n"
        "    print('Hello, world!')\n"
        "\n"
        "def goodbye():\n"
        "    print('Goodbye, world!')\n"
        "\n"
        "class MyClass:\n"
        "    pass\n"
    )

    # src/utils.py
    (src / "utils.py").write_text(
        "import os\n"
        "\n"
        "def helper():\n"
        "    return os.getcwd()\n"
        "\n"
        "CONSTANT = 42\n"
    )

    # docs/readme.md
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "readme.md").write_text(
        "# Project\n"
        "\n"
        "This is a sample project.\n"
        "It has multiple files.\n"
    )

    # data/binary.bin
    data = tmp_path / "data"
    data.mkdir()
    (data / "binary.bin").write_bytes(b"\x00\x01\x02\x03\x04binary content")

    # empty.txt
    (tmp_path / "empty.txt").write_text("")

    # .git/config (should be skipped)
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n    repositoryformatversion = 0\n")

    return tmp_path


class TestGrepTool:
    """Tests for GrepTool."""

    def setup_method(self) -> None:
        """Set up test fixtures.

        We use allow_unrestricted_paths=True so that tests using pytest's
        tmp_path (which lives outside the working directory) are not rejected
        by the path-restriction guard.
        """
        self.tool = GrepTool(allow_unrestricted_paths=True)

    def test_tool_metadata(self) -> None:
        """Test tool has correct name and description."""
        assert self.tool.name == "Search file contents"
        assert "search" in self.tool.description.lower() or "Search" in self.tool.description

    def test_args_schema(self) -> None:
        """Test that args_schema has correct fields and defaults."""
        schema = self.tool.args_schema
        fields = schema.model_fields

        assert "pattern" in fields
        assert fields["pattern"].is_required()

        assert "path" in fields
        assert not fields["path"].is_required()

        assert "glob_pattern" in fields
        assert not fields["glob_pattern"].is_required()

        assert "output_mode" in fields
        assert not fields["output_mode"].is_required()

        assert "case_insensitive" in fields
        assert not fields["case_insensitive"].is_required()

        assert "context_lines" in fields
        assert not fields["context_lines"].is_required()

        assert "include_line_numbers" in fields
        assert not fields["include_line_numbers"].is_required()

    def test_basic_pattern_match(self, sample_dir: Path) -> None:
        """Test simple string pattern found in output."""
        result = self.tool._run(pattern="Hello", path=str(sample_dir))
        assert "Hello" in result

    def test_regex_pattern(self, sample_dir: Path) -> None:
        """Test regex pattern matches function definitions."""
        result = self.tool._run(pattern=r"def\s+\w+", path=str(sample_dir))
        assert "def hello" in result
        assert "def goodbye" in result
        assert "def helper" in result

    def test_case_sensitive_default(self, sample_dir: Path) -> None:
        """Test that search is case-sensitive by default."""
        result = self.tool._run(pattern="hello", path=str(sample_dir))
        # "hello" (lowercase) appears in "def hello():" but not in "Hello, world!"
        assert "hello" in result
        # Verify it found the function definition line
        assert "def hello" in result

    def test_case_insensitive(self, sample_dir: Path) -> None:
        """Test case-insensitive matching."""
        result = self.tool._run(
            pattern="hello", path=str(sample_dir), case_insensitive=True
        )
        # Should match both "def hello():" and "Hello, world!"
        assert "hello" in result.lower()
        assert "Hello" in result

    def test_output_mode_content(self, sample_dir: Path) -> None:
        """Test content output mode shows file paths, line numbers, and text."""
        result = self.tool._run(
            pattern="CONSTANT", path=str(sample_dir), output_mode="content"
        )
        assert "utils.py" in result
        assert "CONSTANT" in result
        # Should have line numbers by default
        assert ": " in result

    def test_output_mode_files_with_matches(self, sample_dir: Path) -> None:
        """Test files_with_matches output mode shows only file paths."""
        result = self.tool._run(
            pattern="def", path=str(sample_dir), output_mode="files_with_matches"
        )
        assert "main.py" in result
        assert "utils.py" in result
        # Should not contain line content
        assert "print" not in result

    def test_output_mode_count(self, sample_dir: Path) -> None:
        """Test count output mode shows filepath: N format."""
        result = self.tool._run(
            pattern="def", path=str(sample_dir), output_mode="count"
        )
        # main.py has 2 def lines, utils.py has 1
        assert "main.py: 2" in result
        assert "utils.py: 1" in result

    def test_context_lines(self, sample_dir: Path) -> None:
        """Test surrounding context lines are included."""
        result = self.tool._run(
            pattern="CONSTANT", path=str(sample_dir), context_lines=2
        )
        # Two lines before CONSTANT = 42 is "    return os.getcwd()"
        assert "return os.getcwd()" in result
        assert "CONSTANT" in result

    def test_line_numbers_disabled(self, sample_dir: Path) -> None:
        """Test output without line number prefixes."""
        result = self.tool._run(
            pattern="CONSTANT",
            path=str(sample_dir),
            include_line_numbers=False,
        )
        assert "CONSTANT = 42" in result
        # Verify no line number prefix (e.g., "6: ")
        for line in result.strip().split("\n"):
            if "CONSTANT" in line:
                assert not line[0].isdigit() or ": " not in line

    def test_glob_pattern_filtering(self, sample_dir: Path) -> None:
        """Test glob pattern filters to specific file types."""
        result = self.tool._run(
            pattern="project",
            path=str(sample_dir),
            glob_pattern="*.py",
            case_insensitive=True,
        )
        # "project" appears in readme.md but not in .py files
        assert "No matches found" in result

    def test_search_single_file(self, sample_dir: Path) -> None:
        """Test searching a single file by path."""
        file_path = str(sample_dir / "src" / "main.py")
        result = self.tool._run(pattern="def", path=file_path)
        assert "def hello" in result
        assert "def goodbye" in result
        # Should not include results from other files
        assert "helper" not in result

    def test_path_not_found(self) -> None:
        """Test error message when a relative path doesn't exist."""
        result = self.tool._run(pattern="test", path="totally_nonexistent_subdir")
        assert "Error" in result
        assert "does not exist" in result

    def test_invalid_regex(self, sample_dir: Path) -> None:
        """Test error message for invalid regex patterns."""
        result = self.tool._run(pattern="[invalid", path=str(sample_dir))
        assert "Error" in result
        assert "Invalid regex" in result

    def test_binary_files_skipped(self, sample_dir: Path) -> None:
        """Test binary files are not included in results."""
        result = self.tool._run(pattern="binary", path=str(sample_dir))
        # binary.bin has null bytes so it should be skipped
        assert "binary.bin" not in result

    def test_no_matches_found(self, sample_dir: Path) -> None:
        """Test message when no matches are found."""
        result = self.tool._run(
            pattern="zzz_nonexistent_pattern_zzz", path=str(sample_dir)
        )
        assert "No matches found" in result

    def test_hidden_dirs_skipped(self, sample_dir: Path) -> None:
        """Test that .git/ directory contents are not searched."""
        result = self.tool._run(pattern="repositoryformatversion", path=str(sample_dir))
        assert "No matches found" in result

    def test_empty_file(self, sample_dir: Path) -> None:
        """Test searching an empty file doesn't crash."""
        result = self.tool._run(
            pattern="anything", path=str(sample_dir / "empty.txt")
        )
        assert "No matches found" in result

    def test_run_with_kwargs(self, sample_dir: Path) -> None:
        """Test _run ignores extra kwargs."""
        result = self.tool._run(
            pattern="Hello", path=str(sample_dir), extra_arg="ignored"
        )
        assert "Hello" in result


class TestPathRestriction:
    """Tests for path traversal prevention and allow_unrestricted_paths."""

    def test_absolute_path_outside_cwd_blocked(self, tmp_path: Path) -> None:
        """An absolute path outside cwd is rejected by default."""
        tool = GrepTool()
        # tmp_path is almost certainly not under os.getcwd()
        result = tool._run(pattern="anything", path=str(tmp_path))
        assert "Error" in result
        assert "outside the working directory" in result

    def test_relative_traversal_blocked(self, sample_dir: Path) -> None:
        """A relative path with ../ that escapes cwd is rejected."""
        tool = GrepTool()
        result = tool._run(pattern="anything", path="../../etc")
        assert "Error" in result
        assert "outside the working directory" in result

    def test_relative_path_within_cwd_allowed(self) -> None:
        """A relative path that stays inside cwd works fine."""
        tool = GrepTool()
        # "." is always within cwd
        result = tool._run(pattern="zzz_will_not_match_anything_zzz", path=".")
        # Should not get a traversal error — either matches or "No matches found"
        assert "outside the working directory" not in result

    def test_allow_unrestricted_paths_bypasses_check(self, tmp_path: Path) -> None:
        """With allow_unrestricted_paths=True, absolute paths outside cwd are allowed."""
        # Write a searchable file in tmp_path
        (tmp_path / "hello.txt").write_text("unrestricted search target\n")
        tool = GrepTool(allow_unrestricted_paths=True)
        result = tool._run(pattern="unrestricted", path=str(tmp_path))
        assert "unrestricted search target" in result

    def test_allow_unrestricted_defaults_false(self) -> None:
        """The flag defaults to False."""
        tool = GrepTool()
        assert tool.allow_unrestricted_paths is False

    def test_error_message_includes_hint(self, tmp_path: Path) -> None:
        """The traversal error tells the user how to opt in."""
        tool = GrepTool()
        result = tool._run(pattern="x", path=str(tmp_path))
        assert "GrepTool(allow_unrestricted_paths=True)" in result


class TestReDoSGuards:
    """Tests for regex denial-of-service mitigations."""

    def test_pattern_length_rejected(self, sample_dir: Path) -> None:
        """Patterns exceeding MAX_REGEX_LENGTH are rejected before compilation."""
        tool = GrepTool(allow_unrestricted_paths=True)
        long_pattern = "a" * (MAX_REGEX_LENGTH + 1)
        result = tool._run(pattern=long_pattern, path=str(sample_dir))
        assert "Error" in result
        assert "Pattern too long" in result

    def test_pattern_at_max_length_accepted(self, sample_dir: Path) -> None:
        """A pattern exactly at MAX_REGEX_LENGTH is allowed (boundary check)."""
        tool = GrepTool(allow_unrestricted_paths=True)
        exact_pattern = "a" * MAX_REGEX_LENGTH
        result = tool._run(pattern=exact_pattern, path=str(sample_dir))
        # Should not get a length error — either matches or "No matches found"
        assert "Pattern too long" not in result

    def test_safe_search_returns_match(self) -> None:
        """_safe_search returns a match object for a normal pattern."""
        compiled = __import__("re").compile(r"hello")
        match = GrepTool._safe_search(compiled, "say hello world")
        assert match is not None
        assert match.group() == "hello"

    def test_safe_search_returns_none_on_no_match(self) -> None:
        """_safe_search returns None when the pattern doesn't match."""
        compiled = __import__("re").compile(r"zzz")
        match = GrepTool._safe_search(compiled, "hello world")
        assert match is None


class TestBraceExpansion:
    """Tests for glob brace expansion ({a,b} syntax)."""

    def test_expand_simple_brace(self) -> None:
        """*.{py,txt} expands to ['*.py', '*.txt']."""
        result = GrepTool._expand_brace_pattern("*.{py,txt}")
        assert result == ["*.py", "*.txt"]

    def test_expand_three_alternatives(self) -> None:
        """*.{py,txt,md} expands to three patterns."""
        result = GrepTool._expand_brace_pattern("*.{py,txt,md}")
        assert result == ["*.py", "*.txt", "*.md"]

    def test_expand_no_braces_passthrough(self) -> None:
        """A pattern without braces is returned as a single-element list."""
        result = GrepTool._expand_brace_pattern("*.py")
        assert result == ["*.py"]

    def test_expand_strips_whitespace(self) -> None:
        """Whitespace around alternatives inside braces is stripped."""
        result = GrepTool._expand_brace_pattern("*.{ py , txt }")
        assert result == ["*.py", "*.txt"]

    def test_expand_prefix_and_suffix(self) -> None:
        """Prefix and suffix around the braces are preserved."""
        result = GrepTool._expand_brace_pattern("src/*.{py,pyi}.bak")
        assert result == ["src/*.py.bak", "src/*.pyi.bak"]

    def test_brace_glob_end_to_end(self, tmp_path: Path) -> None:
        """Brace expansion works end-to-end with _collect_files."""
        (tmp_path / "a.py").write_text("match_me\n")
        (tmp_path / "b.txt").write_text("match_me\n")
        (tmp_path / "c.md").write_text("match_me\n")

        tool = GrepTool(allow_unrestricted_paths=True)
        result = tool._run(
            pattern="match_me",
            path=str(tmp_path),
            glob_pattern="*.{py,txt}",
        )
        assert "a.py" in result
        assert "b.txt" in result
        # .md should NOT be included
        assert "c.md" not in result

    def test_brace_glob_no_duplicates(self, tmp_path: Path) -> None:
        """Files are not reported twice when they match multiple expanded patterns."""
        (tmp_path / "x.py").write_text("unique_content\n")

        tool = GrepTool(allow_unrestricted_paths=True)
        result = tool._run(
            pattern="unique_content",
            path=str(tmp_path),
            glob_pattern="*.{py,py}",
            output_mode="count",
        )
        # Should appear exactly once
        assert result.count("x.py") == 1
