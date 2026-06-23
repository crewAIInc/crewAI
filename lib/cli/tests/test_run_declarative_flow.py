from __future__ import annotations

from pathlib import Path

import pytest

import crewai_cli.run_declarative_flow as run_declarative_flow_module


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
      expr: state.topic
"""


def test_run_declarative_flow_reads_definition_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text(FLOW_YAML, encoding="utf-8")

    run_declarative_flow_module.run_declarative_flow(
        str(definition_path), '{"topic":"AI"}'
    )

    assert capsys.readouterr().out == "AI\n"


def test_run_declarative_flow_rejects_non_object_inputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text(FLOW_YAML, encoding="utf-8")

    with pytest.raises(SystemExit):
        run_declarative_flow_module.run_declarative_flow(
            str(definition_path), '["not", "an", "object"]'
        )

    assert "Invalid --inputs JSON: expected an object." in capsys.readouterr().err


def test_run_declarative_flow_reports_missing_file(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        run_declarative_flow_module.run_declarative_flow("missing-flow.yaml")

    assert (
        "Invalid --definition path: missing-flow.yaml does not exist."
        in capsys.readouterr().err
    )


def test_run_declarative_flow_in_project_env_uses_uv(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    subprocess_calls = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(
        run_declarative_flow_module,
        "build_env_with_all_tool_credentials",
        lambda: {"EXISTING": "value"},
    )
    monkeypatch.setattr(
        run_declarative_flow_module.subprocess,
        "run",
        lambda command, **kwargs: subprocess_calls.append((command, kwargs)),
    )

    run_declarative_flow_module.run_declarative_flow_in_project_env("flow.yaml")

    assert subprocess_calls == [
        (
            ["uv", "run", "crewai", "flow", "kickoff"],
            {
                "capture_output": False,
                "text": True,
                "check": True,
                "env": {"EXISTING": "value"},
            },
        )
    ]


def test_run_declarative_flow_in_process_inside_uv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "flow.yaml").write_text(FLOW_YAML, encoding="utf-8")

    run_declarative_flow_module.run_declarative_flow_in_project_env(
        "flow.yaml", '{"topic":"AI"}'
    )

    assert capsys.readouterr().out == "AI\n"
