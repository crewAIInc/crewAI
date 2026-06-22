from __future__ import annotations

from pathlib import Path

from crewai_cli.create_flow import create_flow


def test_create_flow_declarative_scaffold(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)

    create_flow("Research Flow", declarative=True)

    project_root = tmp_path / "research_flow"
    assert (project_root / "flow.yaml").is_file()
    assert (project_root / "crews").is_dir()
    assert (project_root / "tools").is_dir()
    assert (project_root / "knowledge").is_dir()
    assert (project_root / "skills").is_dir()
    assert not (project_root / "src").exists()

    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'type = "flow"' in pyproject
    assert 'definition = "flow.yaml"' in pyproject
    assert (
        'only-include = ["flow.yaml", "crews", "tools", "knowledge", "skills"]'
        in pyproject
    )

    flow_definition = (project_root / "flow.yaml").read_text(encoding="utf-8")
    assert "schema: crewai.flow/v1" in flow_definition
    assert "name: ResearchFlow" in flow_definition
