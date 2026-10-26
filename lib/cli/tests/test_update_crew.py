"""Tests for ``crewai update``: Poetry-to-uv conversion, and the upgrade hint otherwise."""

from pathlib import Path

from click.testing import CliRunner, Result
import pytest

from crewai_cli.cli import crewai
from crewai_cli.update_crew import migrate_pyproject
from crewai_cli.utils import read_toml


_POETRY_PYPROJECT = """\
[tool.poetry]
name = "demo"
version = "0.1.0"
description = "demo crew"
authors = ["Demo Author <demo@example.com>"]

[tool.poetry.dependencies]
python = ">=3.10,<3.14"
crewai = "^1.0.0"
"""


def _write_project(tmp_path: Path) -> None:
    """Create a minimal Poetry-style project with a lock file in ``tmp_path``."""
    (tmp_path / "pyproject.toml").write_text(_POETRY_PYPROJECT, encoding="utf-8")
    (tmp_path / "poetry.lock").write_text("current lock\n", encoding="utf-8")


def test_migrate_pyproject_backs_up_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The migration renames ``poetry.lock`` to ``poetry-old.lock`` and backs up pyproject."""
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)

    migrate_pyproject("pyproject.toml", "pyproject.toml")

    assert not (tmp_path / "poetry.lock").exists()
    assert (tmp_path / "poetry-old.lock").read_text(encoding="utf-8") == (
        "current lock\n"
    )
    assert (tmp_path / "pyproject-old.toml").exists()


def test_migrate_pyproject_overwrites_existing_lock_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``crewai update`` must replace a stale ``poetry-old.lock``.

    ``os.rename`` refuses to overwrite an existing destination on Windows
    (``FileExistsError``) while POSIX replaces it silently, so re-running the
    migration used to fail only on Windows.
    """
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    (tmp_path / "poetry-old.lock").write_text("stale backup\n", encoding="utf-8")

    migrate_pyproject("pyproject.toml", "pyproject.toml")

    assert not (tmp_path / "poetry.lock").exists()
    assert (tmp_path / "poetry-old.lock").read_text(encoding="utf-8") == (
        "current lock\n"
    )


_UPGRADE_HINT = "crewai install --upgrade-package crewai --upgrade-package crewai-tools"

_UV_PYPROJECT = """\
# keep this comment
[project]
name = "demo"
version = "0.1.0"
requires-python = ">=3.10,<3.14"
dependencies = ["crewai[tools]>=1.0.0,<2.0.0"]

[project.scripts]
demo = "demo.main:run"
run_crew = "demo.main:run"

[build-system]
requires = ["uv_build>=0.8,<0.9"]
build-backend = "uv_build"

[dependency-groups]
dev = ["pytest"]

[tool.crewai]
type = "crew"
"""

_FLOW_DEFINITION_PYPROJECT = """\
[project]
name = "demo_flow"
version = "0.1.0"
requires-python = ">=3.10,<3.14"
dependencies = ["crewai[tools]>=1.0.0,<2.0.0"]

[tool.crewai]
type = "flow"
definition = "flow.json"
"""


def _invoke_update() -> Result:
    return CliRunner().invoke(crewai, ["update"])


@pytest.mark.parametrize(
    "content", [_UV_PYPROJECT, _FLOW_DEFINITION_PYPROJECT], ids=["uv", "flow"]
)
def test_update_leaves_uv_project_untouched_and_prints_upgrade_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """A project already on uv has nothing to convert; the file must not be rewritten."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(content, encoding="utf-8")

    result = _invoke_update()

    assert result.exit_code == 0, result.output
    assert "only converts Poetry projects to uv" in result.output
    assert "does not upgrade CrewAI" in result.output
    assert "uv tool upgrade crewai" in result.output
    assert _UPGRADE_HINT in result.output
    assert (tmp_path / "pyproject.toml").read_text(encoding="utf-8") == content
    assert sorted(p.name for p in tmp_path.iterdir()) == ["pyproject.toml"]


def test_update_without_pyproject_prints_upgrade_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = _invoke_update()

    assert result.exit_code == 0, result.output
    assert "uv tool upgrade crewai" in result.output
    assert _UPGRADE_HINT in result.output
    assert list(tmp_path.iterdir()) == []


def test_update_converts_poetry_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)

    result = _invoke_update()

    assert result.exit_code == 0, result.output
    assert "saved as pyproject-old.toml" in result.output
    assert "Run `crewai install`" in result.output
    assert "uv tool upgrade crewai" not in result.output
    converted = read_toml(tmp_path / "pyproject.toml")
    assert converted["project"]["name"] == "demo"
    assert converted["project"]["dependencies"] == ["crewai>=1.0.0"]
    assert "poetry" not in converted.get("tool", {})
    assert (tmp_path / "pyproject-old.toml").read_text(encoding="utf-8") == (
        _POETRY_PYPROJECT
    )
    assert (tmp_path / "poetry-old.lock").exists()


def test_update_help_says_it_converts_poetry_projects() -> None:
    runner = CliRunner()

    command_help = runner.invoke(crewai, ["update", "--help"])
    top_level_help = runner.invoke(crewai, ["--help"])

    assert command_help.exit_code == 0
    assert "Convert a Poetry project to uv." in command_help.output
    assert "does not upgrade CrewAI" in command_help.output
    assert "uv tool upgrade crewai\n" in command_help.output
    assert f"{_UPGRADE_HINT}\n" in command_help.output
    update_line = next(
        line
        for line in top_level_help.output.splitlines()
        if line.strip().startswith("update ")
    )
    assert "Convert a Poetry project to uv." in update_line
