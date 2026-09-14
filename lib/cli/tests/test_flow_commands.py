from __future__ import annotations

from pathlib import Path
import subprocess

import click
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
    assert (
        "The command 'crewai flow kickoff' is deprecated. Use 'crewai run' instead."
        in result.output
    )
    assert "AI\n" in result.output
    assert "Running the Flow" not in result.output


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
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text(FLOW_YAML, encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.crewai]\ntype = "flow"\ndefinition = " flow.yaml "\n',
        encoding="utf-8",
    )

    from crewai_cli.run_declarative_flow import configured_project_declarative_flow

    assert configured_project_declarative_flow() == definition_path.resolve()


@pytest.mark.parametrize(
    ("definition", "expected_error"),
    [
        ("C:/tmp/flow.yaml", "must be relative to the project root"),
        ("~/flow.yaml", "must be a project-local path"),
        ("../flow.yaml", "must resolve inside the project root"),
    ],
)
def test_configured_project_declarative_flow_rejects_unsafe_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    definition: str,
    expected_error: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.crewai]\ntype = "flow"\ndefinition = "{definition}"\n',
        encoding="utf-8",
    )

    from crewai_cli.run_declarative_flow import configured_project_declarative_flow

    with pytest.raises(click.UsageError) as exc_info:
        configured_project_declarative_flow()

    assert expected_error in exc_info.value.message


def test_configured_project_declarative_flow_allows_normalized_project_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text(FLOW_YAML, encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.crewai]\ntype = "flow"\ndefinition = "src/../flow.yaml"\n',
        encoding="utf-8",
    )

    from crewai_cli.run_declarative_flow import configured_project_declarative_flow

    assert configured_project_declarative_flow() == definition_path.resolve()


def test_configured_project_declarative_flow_rejects_absolute_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    definition = tmp_path / "flow.yaml"
    (tmp_path / "pyproject.toml").write_text(
        f'[tool.crewai]\ntype = "flow"\ndefinition = "{definition.as_posix()}"\n',
        encoding="utf-8",
    )

    from crewai_cli.run_declarative_flow import configured_project_declarative_flow

    with pytest.raises(click.UsageError) as exc_info:
        configured_project_declarative_flow()

    assert "must be relative to the project root" in exc_info.value.message


def test_configured_project_declarative_flow_rejects_symlink_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    outside_definition = tmp_path.parent / "outside-flow.yaml"
    outside_definition.write_text(FLOW_YAML, encoding="utf-8")
    link = tmp_path / "flow.yaml"
    try:
        link.symlink_to(outside_definition)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    (tmp_path / "pyproject.toml").write_text(
        '[tool.crewai]\ntype = "flow"\ndefinition = "flow.yaml"\n',
        encoding="utf-8",
    )

    from crewai_cli.run_declarative_flow import configured_project_declarative_flow

    with pytest.raises(click.UsageError) as exc_info:
        configured_project_declarative_flow()

    assert "must resolve inside the project root" in exc_info.value.message


def test_configured_project_declarative_flow_rejects_missing_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.crewai]\ntype = "flow"\ndefinition = "missing-flow.yaml"\n',
        encoding="utf-8",
    )

    from crewai_cli.run_declarative_flow import configured_project_declarative_flow

    with pytest.raises(click.UsageError) as exc_info:
        configured_project_declarative_flow()

    assert "must point to an existing file" in exc_info.value.message


def test_configured_project_declarative_flow_rejects_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "flow.yaml").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        '[tool.crewai]\ntype = "flow"\ndefinition = "flow.yaml"\n',
        encoding="utf-8",
    )

    from crewai_cli.run_declarative_flow import configured_project_declarative_flow

    with pytest.raises(click.UsageError) as exc_info:
        configured_project_declarative_flow()

    assert "must point to a regular file" in exc_info.value.message
