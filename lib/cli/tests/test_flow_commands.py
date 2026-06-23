from __future__ import annotations

from pathlib import Path
import subprocess

import pytest
from click.testing import CliRunner

from crewai_cli.cli import flow_run
import crewai_cli.plot_flow as plot_flow_module


FLOW_YAML = """\
schema: crewai.flow/v1
name: TestFlow
config:
  suppress_flow_events: true
methods:
  begin:
    start: true
    do:
      call: expression
      expr: "'AI'"
"""


def _write_flow_project(project_root: Path) -> None:
    (project_root / "flow.yaml").write_text(FLOW_YAML, encoding="utf-8")
    (project_root / "pyproject.toml").write_text(
        '[project]\nname = "demo"\n\n'
        '[tool.crewai]\ntype = "flow"\ndefinition = "flow.yaml"\n',
        encoding="utf-8",
    )


def test_flow_kickoff_runs_configured_declarative_definition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _write_flow_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")

    result = CliRunner().invoke(flow_run)

    assert result.exit_code == 0
    assert "DeprecationWarning" in result.output
    assert "Running the Flow\nAI\n" in result.output


def test_plot_flow_runs_configured_declarative_definition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_flow_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")

    plot_flow_module.plot_flow()


def test_flow_kickoff_delegates_to_run_crew(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    monkeypatch.setattr(
        "crewai_cli.cli.run_crew",
        lambda **kwargs: calls.append(kwargs),
    )

    result = CliRunner().invoke(flow_run)

    assert result.exit_code == 0
    assert calls == [
        {"trained_agents_file": None, "definition": None, "inputs": None},
    ]


def test_plot_flow_keeps_python_entrypoint_without_definition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    subprocess_calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess_calls.append((command, kwargs)),
    )

    plot_flow_module.plot_flow()

    assert subprocess_calls == [
        (
            ["uv", "run", "plot"],
            {"capture_output": False, "text": True, "check": True},
        )
    ]


def test_configured_project_declarative_flow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.crewai]\ntype = "flow"\ndefinition = " flow.yaml "\n',
        encoding="utf-8",
    )

    from crewai_cli.run_declarative_flow import configured_project_declarative_flow

    assert configured_project_declarative_flow() == "flow.yaml"
