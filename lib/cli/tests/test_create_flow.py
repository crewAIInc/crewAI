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
    agents_md = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "CrewAI Flow declaration" in agents_md
    assert "schema: crewai.flow/v1" in agents_md
    assert "do not assemble the string with CEL `+`" in agents_md
    assert "Agent prompt template. Insert Flow values with `${...}`" in agents_md
    assert "Runtime inputs passed to the Crew" in agents_md
    assert "call: expression" in agents_md
    assert "call: tool" not in agents_md
    assert "call: script" not in agents_md
    assert "call: each" not in agents_md
    assert "human_feedback" not in agents_md
    claude_md = (project_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude_md.splitlines()
    gemini_md = (project_root / "GEMINI.md").read_text(encoding="utf-8")
    assert "@./AGENTS.md" in gemini_md.splitlines()

    monkeypatch.chdir(project_root)
    result = CliRunner().invoke(crewai, ["run"], env={"UV_RUN_RECURSION_DEPTH": "1"})

    assert result.exit_code == 0
    assert "Running the Flow" not in result.output
    assert "AI agents" in result.output


def test_create_flow_scaffolds_assistant_instructions(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow")

    project_root = tmp_path / "research_flow"
    agents_md = (project_root / "AGENTS.md").read_text(encoding="utf-8")
    assert "CrewAI Reference for AI Coding Assistants" in agents_md
    assert "Never disable, block, or silence CrewAI's built-in observability" in agents_md
    claude_md = (project_root / "CLAUDE.md").read_text(encoding="utf-8")
    assert "@AGENTS.md" in claude_md.splitlines()
    gemini_md = (project_root / "GEMINI.md").read_text(encoding="utf-8")
    assert "@./AGENTS.md" in gemini_md.splitlines()


def test_create_flow_writes_utf8_under_cp1252_locale(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    import builtins

    real_open = builtins.open

    def cp1252_default_open(file, mode="r", *args, **kwargs):
        if "b" not in mode and "encoding" not in kwargs and not args:
            kwargs["encoding"] = "cp1252"
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", cp1252_default_open)
    monkeypatch.chdir(tmp_path)
    create_flow("Research Flow")

    tasks_yaml = (
        tmp_path
        / "research_flow"
        / "src"
        / "research_flow"
        / "crews"
        / "content_crew"
        / "config"
        / "tasks.yaml"
    ).read_text(encoding="utf-8")
    assert "—" in tasks_yaml
