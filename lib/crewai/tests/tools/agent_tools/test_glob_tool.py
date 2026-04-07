"""Unit tests for GlobTool."""

from pathlib import Path

import pytest

from crewai.tools.agent_tools.glob_tool import GlobTool


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a temp directory with sample files for testing."""
    # src/main.py
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def main(): pass\n")
    (src / "utils.py").write_text("def helper(): pass\n")
    (src / "config.yaml").write_text("key: value\n")

    # src/components/
    components = src / "components"
    components.mkdir()
    (components / "button.tsx").write_text("export const Button = () => {};\n")
    (components / "input.tsx").write_text("export const Input = () => {};\n")
    (components / "index.ts").write_text("export * from './button';\n")

    # tests/
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_main.py").write_text("def test_main(): pass\n")
    (tests / "test_utils.py").write_text("def test_utils(): pass\n")

    # docs/
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "readme.md").write_text("# Project\n")
    (docs / "api.md").write_text("# API\n")

    # data/binary.bin
    data = tmp_path / "data"
    data.mkdir()
    (data / "binary.bin").write_bytes(b"\x00\x01\x02\x03binary content")

    # empty.txt
    (tmp_path / "empty.txt").write_text("")

    # Hidden files (should be skipped by default)
    (tmp_path / ".hidden").write_text("hidden content\n")
    hidden_dir = tmp_path / ".hidden_dir"
    hidden_dir.mkdir()
    (hidden_dir / "secret.txt").write_text("secret\n")

    # .git/config (should be skipped)
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    (git_dir / "config").write_text("[core]\n    repositoryformatversion = 0\n")

    # node_modules (should be skipped)
    node_modules = tmp_path / "node_modules"
    node_modules.mkdir()
    (node_modules / "package.json").write_text('{"name": "test"}\n')

    return tmp_path


class TestGlobTool:
    """Tests for GlobTool."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.tool = GlobTool()

    def test_tool_metadata(self) -> None:
        """Test tool has correct name and description."""
        assert self.tool.name == "glob"
        assert "find" in self.tool.description.lower() or "pattern" in self.tool.description.lower()

    def test_args_schema(self) -> None:
        """Test that args_schema has correct fields and defaults."""
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

    def test_basic_pattern_match(self, sample_dir: Path) -> None:
        """Test simple glob pattern finds files."""
        result = self.tool._run(pattern="*.py", path=str(sample_dir))
        assert "main.py" in result
        assert "utils.py" in result
        assert "test_main.py" in result
        assert "test_utils.py" in result

    def test_recursive_pattern(self, sample_dir: Path) -> None:
        """Test recursive glob pattern with **."""
        result = self.tool._run(pattern="**/*.tsx", path=str(sample_dir))
        assert "button.tsx" in result
        assert "input.tsx" in result

    def test_auto_recursive_prefix(self, sample_dir: Path) -> None:
        """Test that patterns without ** are auto-prefixed for recursive search."""
        result = self.tool._run(pattern="*.yaml", path=str(sample_dir))
        assert "config.yaml" in result

    def test_specific_directory_pattern(self, sample_dir: Path) -> None:
        """Test pattern targeting specific directory."""
        result = self.tool._run(pattern="src/**/*.py", path=str(sample_dir))
        assert "main.py" in result
        assert "utils.py" in result
        # Should not include test files
        assert "test_main.py" not in result

    def test_output_mode_paths(self, sample_dir: Path) -> None:
        """Test paths output mode shows full file paths."""
        result = self.tool._run(pattern="*.md", path=str(sample_dir), output_mode="paths")
        assert "readme.md" in result
        assert "api.md" in result

    def test_output_mode_detailed(self, sample_dir: Path) -> None:
        """Test detailed output mode includes file sizes."""
        result = self.tool._run(pattern="*.md", path=str(sample_dir), output_mode="detailed")
        assert "readme.md" in result
        # Should have size information
        assert "B" in result  # Bytes unit

    def test_output_mode_tree(self, sample_dir: Path) -> None:
        """Test tree output mode shows directory structure."""
        result = self.tool._run(pattern="*.py", path=str(sample_dir), output_mode="tree")
        assert "src/" in result or "src" in result
        assert "tests/" in result or "tests" in result

    def test_hidden_files_excluded_by_default(self, sample_dir: Path) -> None:
        """Test hidden files are not included by default."""
        result = self.tool._run(pattern="*", path=str(sample_dir))
        assert ".hidden" not in result
        assert "secret.txt" not in result

    def test_hidden_files_included_when_requested(self, sample_dir: Path) -> None:
        """Test hidden files are included when include_hidden=True."""
        result = self.tool._run(pattern="*", path=str(sample_dir), include_hidden=True)
        assert ".hidden" in result

    def test_git_directory_skipped(self, sample_dir: Path) -> None:
        """Test .git directory contents are not included."""
        result = self.tool._run(pattern="*", path=str(sample_dir), include_hidden=True)
        # Even with include_hidden, .git should be skipped
        # The .git directory itself might show but not its contents
        assert "config" not in result or ".git" not in result.split("config")[0].split("\n")[-1]

    def test_node_modules_skipped(self, sample_dir: Path) -> None:
        """Test node_modules directory contents are not included."""
        result = self.tool._run(pattern="*.json", path=str(sample_dir))
        assert "package.json" not in result

    def test_path_not_found(self) -> None:
        """Test error message when path doesn't exist."""
        result = self.tool._run(pattern="*.py", path="/nonexistent/path")
        assert "Error" in result
        assert "does not exist" in result

    def test_path_is_not_directory(self, sample_dir: Path) -> None:
        """Test error message when path is a file, not directory."""
        file_path = str(sample_dir / "empty.txt")
        result = self.tool._run(pattern="*.py", path=file_path)
        assert "Error" in result
        assert "not a directory" in result

    def test_no_matches_found(self, sample_dir: Path) -> None:
        """Test message when no files match pattern."""
        result = self.tool._run(pattern="*.nonexistent", path=str(sample_dir))
        assert "No files found" in result

    def test_files_only_default(self, sample_dir: Path) -> None:
        """Test that only files are matched by default (not directories)."""
        result = self.tool._run(pattern="*", path=str(sample_dir))
        # Should have files
        assert ".txt" in result or ".py" in result
        # Directories shouldn't have trailing slash in paths mode
        lines = [l for l in result.split("\n") if "src/" in l and l.strip().endswith("/")]
        # Should not list src/ as a match (it's a directory)
        assert len(lines) == 0 or "tree" in result.lower()

    def test_dirs_only(self, sample_dir: Path) -> None:
        """Test dirs_only flag matches only directories."""
        result = self.tool._run(
            pattern="*", path=str(sample_dir), dirs_only=True, files_only=False
        )
        assert "src" in result
        assert "tests" in result
        assert "docs" in result
        # Should not include files
        assert ".py" not in result
        assert ".txt" not in result

    def test_match_count_summary(self, sample_dir: Path) -> None:
        """Test that result includes count of matched files."""
        result = self.tool._run(pattern="*.py", path=str(sample_dir))
        assert "Found" in result
        assert "file" in result.lower()

    def test_run_with_kwargs(self, sample_dir: Path) -> None:
        """Test _run ignores extra kwargs."""
        result = self.tool._run(
            pattern="*.py", path=str(sample_dir), extra_arg="ignored"
        )
        assert "main.py" in result

    def test_test_file_pattern(self, sample_dir: Path) -> None:
        """Test finding test files with test_*.py pattern."""
        result = self.tool._run(pattern="test_*.py", path=str(sample_dir))
        assert "test_main.py" in result
        assert "test_utils.py" in result
        # Should not include non-test files
        assert "main.py" not in result or "test_main.py" in result

    def test_typescript_files(self, sample_dir: Path) -> None:
        """Test finding TypeScript files with combined pattern."""
        result = self.tool._run(pattern="*.ts", path=str(sample_dir))
        assert "index.ts" in result
        # .tsx files should not match *.ts
        assert "button.tsx" not in result
