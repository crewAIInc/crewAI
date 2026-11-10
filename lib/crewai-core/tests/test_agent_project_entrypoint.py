from pathlib import Path
from typing import Any

from crewai_core.project import ProjectDefinitionError, resolve_agent_project_entrypoint
import pytest


def project_config(root: Path, definition: str | None = None) -> None:
    config = '[project]\nname = "support_agent"\n[tool.crewai]\ntype = "agent"\n'
    if definition is not None:
        config += f'definition = "{definition}"\n'
    (root / "pyproject.toml").write_text(config)


@pytest.mark.parametrize("suffix", ["py", "json", "yml", "yaml"])
def test_discovers_one_root_agent_definition(tmp_path: Path, suffix: str) -> None:
    project_config(tmp_path)
    definition = tmp_path / f"agent.{suffix}"
    definition.write_text("")
    assert resolve_agent_project_entrypoint(tmp_path) == definition


def test_ambiguous_agent_definitions_require_explicit_choice(tmp_path: Path) -> None:
    project_config(tmp_path)
    (tmp_path / "agent.json").write_text("{}")
    (tmp_path / "agent.yaml").write_text("{}")
    with pytest.raises(ProjectDefinitionError, match="Multiple agent definitions"):
        resolve_agent_project_entrypoint(tmp_path)
    project_config(tmp_path, "agent.yaml")
    assert resolve_agent_project_entrypoint(tmp_path) == tmp_path / "agent.yaml"


@pytest.mark.parametrize("definition", ["missing.json", "agent.txt", "../outside.json"])
def test_rejects_invalid_explicit_agent_definition(
    tmp_path: Path, definition: str
) -> None:
    project_config(tmp_path, definition)
    (tmp_path / "agent.txt").write_text("{}")
    with pytest.raises(ProjectDefinitionError):
        resolve_agent_project_entrypoint(tmp_path)


@pytest.mark.parametrize("explicit", [True, False])
def test_agent_symlink_cannot_escape_project(tmp_path: Path, explicit: bool) -> None:
    root = tmp_path / "project"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    (root / "agent.json").symlink_to(outside)
    project_config(root, "agent.json" if explicit else None)
    with pytest.raises(ProjectDefinitionError, match="inside the project root"):
        resolve_agent_project_entrypoint(root)


@pytest.mark.parametrize("name", ["Support-Agent", "support agent", "Support.Agent"])
def test_normalizes_supplied_package_name(tmp_path: Path, name: str) -> None:
    project_config(tmp_path)
    package = "supportagent" if name == "Support.Agent" else "support_agent"
    definition = tmp_path / "src" / package / "agent.py"
    definition.parent.mkdir(parents=True)
    definition.write_text("")
    assert resolve_agent_project_entrypoint(tmp_path, name) == definition


@pytest.mark.parametrize("name", [42, "class", "../support_agent", "some\\path", ""])
def test_rejects_invalid_supplied_package_name(tmp_path: Path, name: object) -> None:
    project_config(tmp_path)
    with pytest.raises(ProjectDefinitionError, match="Python package name"):
        resolve_agent_project_entrypoint(tmp_path, name)  # type: ignore[arg-type]


def test_entrypoint_resolution_never_imports_agent_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    project_config(tmp_path)
    definition = tmp_path / "agent.json"
    definition.write_text("{}")
    original_import = builtins.__import__

    def import_without_sdk(name: str, *args: Any, **kwargs: Any) -> object:
        if name == "crewai" or name.startswith("crewai."):
            raise AssertionError("Entrypoint resolution imported the SDK")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_sdk)
    assert resolve_agent_project_entrypoint(tmp_path) == definition


def test_missing_agent_definition_has_actionable_error(tmp_path: Path) -> None:
    project_config(tmp_path)
    with pytest.raises(ProjectDefinitionError, match="No agent definition found"):
        resolve_agent_project_entrypoint(tmp_path)
