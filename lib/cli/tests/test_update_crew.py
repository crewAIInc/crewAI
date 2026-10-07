"""Tests for the ``crewai update`` pyproject migration."""

import tomllib
from pathlib import Path

import pytest

from crewai_cli.update_crew import migrate_pyproject


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

_LEGACY_PYPROJECT = """\
[tool.poetry]
name = "legacy"
version = "0.2.0"
description = "legacy crew"
authors = ["Jane Doe"]

[tool.poetry.dependencies]
crewai = "^1.0.0"
httpx = { version = "^0.27", optional = true }
requests = "2.31.0"
starlette = "*"
jinja2 = "~3.1.2"

[tool.poetry.scripts]
serve = "serve"
"""


def _write_project(tmp_path: Path) -> None:
    """Create a minimal Poetry-style project with a lock file in ``tmp_path``."""
    (tmp_path / "pyproject.toml").write_text(_POETRY_PYPROJECT, encoding="utf-8")
    (tmp_path / "poetry.lock").write_text("current lock\n", encoding="utf-8")


def _write_legacy_project(tmp_path: Path) -> None:
    """Create a Poetry project exercising the legacy-format edge cases."""
    (tmp_path / "pyproject.toml").write_text(_LEGACY_PYPROJECT, encoding="utf-8")
    (tmp_path / "poetry.lock").write_text("current lock\n", encoding="utf-8")


def _migrated_project(tmp_path: Path) -> dict:
    return tomllib.loads((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))


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


def test_migrate_pyproject_accepts_name_only_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poetry author entry without an email must not crash the migration."""
    monkeypatch.chdir(tmp_path)
    _write_legacy_project(tmp_path)

    migrate_pyproject("pyproject.toml", "pyproject.toml")

    project = _migrated_project(tmp_path)["project"]
    assert project["authors"] == [{"name": "Jane Doe"}]


def test_migrate_pyproject_omits_requires_python_without_poetry_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A poetry project without a python constraint must not produce a None
    ``requires-python`` (``tomli_w`` cannot serialize None)."""
    monkeypatch.chdir(tmp_path)
    _write_legacy_project(tmp_path)

    migrate_pyproject("pyproject.toml", "pyproject.toml")

    project = _migrated_project(tmp_path)["project"]
    assert "requires-python" not in project


def test_migrate_pyproject_translates_dependency_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Poetry constraint forms must become valid PEP 508 requirements."""
    monkeypatch.chdir(tmp_path)
    _write_legacy_project(tmp_path)

    migrate_pyproject("pyproject.toml", "pyproject.toml")

    dependencies = _migrated_project(tmp_path)["project"]["dependencies"]
    assert "crewai>=1.0.0" in dependencies
    assert "httpx>=0.27" in dependencies, "a table dependency without extras must not produce 'httpx[]'"
    assert "requests==2.31.0" in dependencies, "a bare poetry version is an exact pin"
    assert "starlette" in dependencies, "'*' means any version"
    assert "jinja2>=3.1.2,<3.2" in dependencies


def test_migrate_pyproject_survives_scripts_without_dotted_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scripts whose values contain no module path must not crash the
    ``run_crew`` injection."""
    monkeypatch.chdir(tmp_path)
    _write_legacy_project(tmp_path)

    migrate_pyproject("pyproject.toml", "pyproject.toml")

    assert _migrated_project(tmp_path)["project"]["scripts"] == {"serve": "serve"}
