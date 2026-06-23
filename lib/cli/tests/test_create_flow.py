from __future__ import annotations

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

    monkeypatch.chdir(project_root)
    result = CliRunner().invoke(
        crewai, ["flow", "kickoff"], env={"UV_RUN_RECURSION_DEPTH": "1"}
    )

    assert result.exit_code == 0
    assert "Running the Flow" in result.output
    assert "AI agents" in result.output
