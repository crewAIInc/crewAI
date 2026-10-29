"""Tests for ``crewai update``: Poetry-to-uv conversion, and the upgrade hint otherwise."""

from pathlib import Path

from crewai_cli.update_crew import migrate_pyproject, parse_version
from packaging.requirements import Requirement
import pytest
import tomli
from click.testing import CliRunner, Result

from crewai_cli.cli import crewai
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
    return tomli.loads((tmp_path / "pyproject.toml").read_text(encoding="utf-8"))


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
    assert "httpx>=0.27" in dependencies, (
        "a table dependency without extras must not produce 'httpx[]'"
    )
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


@pytest.mark.parametrize(
    "version, expected",
    [
        ("*", ""),
        ("~1", ">=1,<2"),
        ("~1.2", ">=1.2,<1.3"),
        ("~1.2.3", ">=1.2.3,<1.3"),
        ("~0.2", ">=0.2,<0.3"),
        ("~1.2rc1", ">=1.2rc1,<1.3"),
        ("~1.2.3rc1", ">=1.2.3rc1,<1.3"),
        ("~1!1.2", ">=1!1.2,<1!1.3"),
        ("~=1.2", "~=1.2"),
        ("1.2", "==1.2"),
        (">=1.2,<2", ">=1.2,<2"),
    ],
)
def test_parse_version_preserves_poetry_constraints(
    version: str, expected: str
) -> None:
    assert parse_version(version) == expected


@pytest.mark.parametrize("version", ["", "  "])
@pytest.mark.parametrize("table_dependency", [False, True])
def test_migration_rejects_empty_constraints_without_changing_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    table_dependency: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    dependency = f'{{ version = "{version}" }}' if table_dependency else f'"{version}"'
    path = tmp_path / "pyproject.toml"
    path.write_text(_POETRY_PYPROJECT + f"httpx = {dependency}\n", encoding="utf-8")

    original = path.read_bytes()
    with pytest.raises(ValueError, match="Poetry version constraint must not be empty"):
        migrate_pyproject("pyproject.toml", "pyproject.toml")

    assert path.read_bytes() == original
    assert (tmp_path / "poetry.lock").read_text(encoding="utf-8") == "current lock\n"
    assert not (tmp_path / "poetry-old.lock").exists()
    assert not (tmp_path / "pyproject-old.toml").exists()


@pytest.mark.parametrize("version", ["~1.2", "~1.2.3", "~1.2rc1"])
def test_migration_preserves_tilde_minor_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    path = tmp_path / "pyproject.toml"
    path.write_text(_POETRY_PYPROJECT + f'httpx = "{version}"\n', encoding="utf-8")

    migrate_pyproject("pyproject.toml", "pyproject.toml")

    dependencies = _migrated_project(tmp_path)["project"]["dependencies"]
    requirement = Requirement(
        next(dep for dep in dependencies if dep.startswith("httpx"))
    )
    assert requirement.specifier.contains(version[1:], prereleases=True)
    assert requirement.specifier.contains("1.2.9", prereleases=True)
    assert not requirement.specifier.contains("1.3.0", prereleases=True)


@pytest.mark.parametrize(
    "version",
    ["~1.2, !=1.2.5", "!=1.2.5, ~1.2", "~1.2, >=1.2.3, !=1.2.5"],
)
@pytest.mark.parametrize("table_dependency", [False, True])
def test_migration_preserves_compound_tilde_constraints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str,
    table_dependency: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_project(tmp_path)
    dependency = f'{{ version = "{version}" }}' if table_dependency else f'"{version}"'
    path = tmp_path / "pyproject.toml"
    path.write_text(_POETRY_PYPROJECT + f"httpx = {dependency}\n", encoding="utf-8")

    migrate_pyproject("pyproject.toml", "pyproject.toml")

    dependencies = _migrated_project(tmp_path)["project"]["dependencies"]
    requirement = Requirement(
        next(dep for dep in dependencies if dep.startswith("httpx"))
    )
    assert requirement.specifier.contains("1.2.3")
    assert requirement.specifier.contains("1.2.6")
    assert not requirement.specifier.contains("1.2.5")
    assert not requirement.specifier.contains("1.1.9")
    assert not requirement.specifier.contains("1.3.0")
    assert requirement.specifier.contains("1.2.2") == (">=1.2.3" not in version)
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

# Poetry 2 projects can use the PEP 621 layout with no [tool.poetry] table; uv
# already reads them, so there is nothing to convert.
_POETRY2_PEP621_PYPROJECT = """\
[project]
name = "demo"
version = "0.1.0"
requires-python = ">=3.10,<3.14"
dependencies = ["crewai>=1.0.0"]

[build-system]
requires = ["poetry-core>=2.0.0,<3.0.0"]
build-backend = "poetry.core.masonry.api"
"""

_NOTHING_TO_CONVERT = "only converts projects with a [tool.poetry] table to uv"


def _invoke_update() -> Result:
    return CliRunner().invoke(crewai, ["update"])


@pytest.mark.parametrize(
    "content",
    [_UV_PYPROJECT, _FLOW_DEFINITION_PYPROJECT, _POETRY2_PEP621_PYPROJECT],
    ids=["uv", "flow", "poetry2-pep621"],
)
def test_update_leaves_uv_project_untouched_and_prints_upgrade_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    """A project already on uv has nothing to convert; the file must not be rewritten."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(content, encoding="utf-8")

    result = _invoke_update()

    assert result.exit_code == 0, result.output
    assert _NOTHING_TO_CONVERT in result.output
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
    assert _NOTHING_TO_CONVERT in result.output
    assert "does not upgrade CrewAI" in result.output
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
