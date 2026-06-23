from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import subprocess

import pytest

import crewai_cli.kickoff_flow as kickoff_flow_module
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


def test_kickoff_flow_runs_configured_declarative_definition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_flow_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")

    kickoff_flow_module.kickoff_flow()

    assert capsys.readouterr().out == "AI\n"


def test_plot_flow_runs_configured_declarative_definition(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_flow_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")

    plot_flow_module.plot_flow()


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        pytest.param(kickoff_flow_module.kickoff_flow, ["uv", "run", "kickoff"]),
        pytest.param(plot_flow_module.plot_flow, ["uv", "run", "plot"]),
    ],
)
def test_flow_commands_keep_python_entrypoint_without_definition(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    command: Callable[[], None],
    expected: list[str],
) -> None:
    subprocess_calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess_calls.append((command, kwargs)),
    )

    command()

    assert subprocess_calls == [
        (
            expected,
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
