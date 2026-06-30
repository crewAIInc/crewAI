from __future__ import annotations

import builtins
from pathlib import Path

from click.testing import CliRunner
from pytest import MonkeyPatch
import tomli

from crewai_cli.cli import crewai
from crewai_cli.create_flow import create_flow


def test_create_flow_declarative_project_can_run(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow", declarative=True)

    project_root = tmp_path / "research_flow"
    assert project_root.is_dir()

    pyproject = tomli.loads(
        (project_root / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["name"] == "research_flow"
    assert pyproject["project"]["requires-python"]
    assert pyproject["project"]["dependencies"]
    assert (project_root / pyproject["tool"]["crewai"]["definition"]).is_file()
    agents_md = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "CrewAI Flow declaration" in agents_md
    assert "schema: crewai.flow/v1" in agents_md

    monkeypatch.chdir(project_root)
    result = CliRunner().invoke(crewai, ["run"], env={"UV_RUN_RECURSION_DEPTH": "1"})

    assert result.exit_code == 0
    assert "Running the Flow" not in result.output
    assert "AI agents" in result.output


def test_create_flow_declarative_project_scaffolds_without_crewai_framework(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)

    real_import = builtins.__import__

    def block_crewai_framework_import(
        name: str,
        globals=None,
        locals=None,
        fromlist=(),
        level: int = 0,
    ):
        if name == "crewai" or name.startswith("crewai."):
            raise ImportError("standalone crewai-cli install has no crewai package")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_crewai_framework_import)

    create_flow("Standalone Flow", declarative=True)

    project_root = tmp_path / "standalone_flow"
    assert (project_root / "AGENTS.md").is_file()
    assert (project_root / "src" / "standalone_flow" / "flow.yaml").is_file()
