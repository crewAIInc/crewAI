from pathlib import Path
import subprocess
import tempfile

import pytest


@pytest.fixture
def temp_project():
    temp_dir = tempfile.TemporaryDirectory()
    project_dir = Path(temp_dir.name) / "test_project"
    project_dir.mkdir()

    pyproject_content = """
    [project]
    name = "test-project"
    version = "0.1.0"
    description = "Test project"
    requires-python = ">=3.10"
    """

    (project_dir / "pyproject.toml").write_text(pyproject_content)
    run_command(
        ["uv", "add", "--editable", f"file://{Path.cwd().absolute()}"], project_dir
    )
    run_command(["uv", "sync"], project_dir)
    yield project_dir


def run_command(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


@pytest.mark.skip(reason="Test takes too long in GitHub Actions (>30s timeout) due to dependency installation")
def test_no_optional_dependencies_in_init(temp_project):
    """
    Test that crewai-tools can be imported without optional dependencies.

    The package defines optional dependencies in pyproject.toml, but the base
    package should be importable without any of these optional dependencies
    being installed.
    """
    result = run_command(
        ["uv", "run", "python", "-c", "import crewai_tools"], temp_project
    )
    assert result.returncode == 0, f"Import failed with error: {result.stderr}"
