"""Unit tests for GlobTool."""

from pathlib import Path

import pytest

from crewai_tools import GlobTool


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample files for testing."""
    # src/main.py
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("print('hello')")
    (src / "utils.py").write_text("def helper(): pass")

    # src/components/button.tsx
    components = src / "components"
    components.mkdir()
    (components / "button.tsx").write_text("export const Button = () => {}")
    (components / "input.tsx").write_text("export const Input = () => {}")

    # tests/test_main.py
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("def test_hello(): pass")
    (tests / "test_utils.py").write_text("def test_helper(): pass")

    # config files
    (tmp_path / "config.yaml").write_text("key: value")
    (tmp_path / "settings.json").write_text("{}")

    # hidden file
    (tmp_path / ".hidden").write_text("secret")

    # empty directory
    (tmp_path / "empty_dir").mkdir()

    return tmp_path


class TestGlobTool:
    """Tests for GlobTool."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.tool = GlobTool()

    def test_tool_metadata(self) -> None:
        """Test tool has correct name and description."""
        assert self.tool.name == "glob"
        assert "pattern" in self.tool.description.lower()

    def test_args_schema(self) -> None:
        """Test that args_schema has correct fields."""
        schema = self.tool.args_schema
        fields = schema.model_fields

        assert "pattern" in fields
        assert fields["pattern"].is_required()

        assert "path" in fields
        assert not fields["path"].is_required()

        assert "output_mode" in fields
        assert not fields["output_mode"].is_required()

        assert "include_hidden" in fields
        assert not fields["include_hidden"].is_required()

    def test_find_python_files(self, sample_dir: Path) -> None:
        """Test finding Python files with *.py pattern."""
        result = self.tool._run(pattern="*.py", path=str(sample_dir))
        assert "main.py" in result
        assert "utils.py" in result
        assert "test_main.py" in result
        assert "test_utils.py" in result

    def test_find_specific_extension(self, sample_dir: Path) -> None:
        """Test finding files with specific extension."""
        result = self.tool._run(pattern="*.tsx", path=str(sample_dir))
        assert "button.tsx" in result
        assert "input.tsx" in result
        assert "main.py" not in result

    def test_find_test_files(self, sample_dir: Path) -> None:
        """Test finding test files with test_*.py pattern."""
        result = self.tool._run(pattern="test_*.py", path=str(sample_dir))
        assert "test_main.py" in result
        assert "test_utils.py" in result
        # Verify non-test files are not included (check for exact filename, not substring)
        lines = result.split("\n")
        file_lines = [l for l in lines if l.endswith(".py")]
        assert not any(l.endswith("/main.py") or l == "main.py" for l in file_lines)
        assert not any(l.endswith("/utils.py") or l == "utils.py" for l in file_lines)

    def test_recursive_pattern(self, sample_dir: Path) -> None:
        """Test explicit recursive pattern **/*.py."""
        result = self.tool._run(pattern="**/*.py", path=str(sample_dir))
        assert "main.py" in result
        assert "test_main.py" in result

    def test_hidden_files_excluded_by_default(self, sample_dir: Path) -> None:
        """Test hidden files are excluded by default."""
        result = self.tool._run(pattern="*", path=str(sample_dir))
        assert ".hidden" not in result

    def test_hidden_files_included(self, sample_dir: Path) -> None:
        """Test hidden files included when include_hidden=True."""
        result = self.tool._run(
            pattern=".*", path=str(sample_dir), include_hidden=True
        )
        assert ".hidden" in result

    def test_output_mode_paths(self, sample_dir: Path) -> None:
        """Test paths output mode shows file paths."""
        result = self.tool._run(
            pattern="*.py", path=str(sample_dir), output_mode="paths"
        )
        # Should contain full paths
        assert str(sample_dir) in result or "main.py" in result

    def test_output_mode_detailed(self, sample_dir: Path) -> None:
        """Test detailed output mode shows sizes."""
        result = self.tool._run(
            pattern="*.py", path=str(sample_dir), output_mode="detailed"
        )
        # Should contain size indicators
        assert "B" in result  # Bytes indicator

    def test_output_mode_tree(self, sample_dir: Path) -> None:
        """Test tree output mode shows directory structure."""
        result = self.tool._run(
            pattern="*.py", path=str(sample_dir), output_mode="tree"
        )
        assert str(sample_dir) in result

    def test_dirs_only(self, sample_dir: Path) -> None:
        """Test dirs_only=True only returns directories."""
        result = self.tool._run(
            pattern="*",
            path=str(sample_dir),
            dirs_only=True,
            files_only=False,
        )
        assert "src" in result or "tests" in result or "empty_dir" in result
        # Should not contain file extensions
        assert ".py" not in result
        assert ".yaml" not in result

    def test_path_not_found(self) -> None:
        """Test error message when path doesn't exist."""
        result = self.tool._run(pattern="*.py", path="/nonexistent/path")
        assert "Error" in result
        assert "does not exist" in result

    def test_path_is_file(self, sample_dir: Path) -> None:
        """Test error message when path is a file, not directory."""
        file_path = sample_dir / "config.yaml"
        result = self.tool._run(pattern="*.py", path=str(file_path))
        assert "Error" in result
        assert "not a directory" in result

    def test_no_matches(self, sample_dir: Path) -> None:
        """Test message when no files match pattern."""
        result = self.tool._run(pattern="*.xyz", path=str(sample_dir))
        assert "No files found" in result

    def test_found_count_in_output(self, sample_dir: Path) -> None:
        """Test that result includes count of found files."""
        result = self.tool._run(pattern="*.py", path=str(sample_dir))
        assert "Found" in result
        assert "file(s)" in result

    def test_run_with_kwargs(self, sample_dir: Path) -> None:
        """Test _run ignores extra kwargs."""
        result = self.tool._run(
            pattern="*.py", path=str(sample_dir), extra_arg="ignored"
        )
        assert "main.py" in result
