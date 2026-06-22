from __future__ import annotations

from pathlib import Path

import crewai_cli.kickoff_flow as kickoff_flow_module
import crewai_cli.plot_flow as plot_flow_module


def test_kickoff_flow_uses_configured_definition(monkeypatch):
    calls = []
    subprocess_calls = []
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)

    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.configured_project_declarative_flow",
        lambda: "flow.yaml",
    )
    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.run_declarative_flow",
        lambda **kwargs: calls.append({"in_process": kwargs}),
    )
    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.run_declarative_flow_in_project_env",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        kickoff_flow_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    kickoff_flow_module.kickoff_flow()

    assert calls == [{"definition": "flow.yaml"}]
    assert subprocess_calls == []


def test_kickoff_flow_keeps_python_entrypoint_without_definition(monkeypatch):
    subprocess_calls = []

    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.configured_project_declarative_flow",
        lambda: None,
    )
    monkeypatch.setattr(
        kickoff_flow_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    kickoff_flow_module.kickoff_flow()

    assert subprocess_calls == [
        (
            (["uv", "run", "kickoff"],),
            {"capture_output": False, "text": True, "check": True},
        )
    ]


def test_plot_flow_uses_configured_definition(monkeypatch):
    calls = []
    subprocess_calls = []
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)

    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.configured_project_declarative_flow",
        lambda: "flow.yaml",
    )
    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.plot_declarative_flow",
        lambda definition: calls.append({"in_process": definition}),
    )
    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.plot_declarative_flow_in_project_env",
        lambda definition: calls.append(definition),
    )
    monkeypatch.setattr(
        plot_flow_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    plot_flow_module.plot_flow()

    assert calls == ["flow.yaml"]
    assert subprocess_calls == []


def test_plot_flow_delegates_project_env_detection_to_runner(monkeypatch):
    calls = []
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.configured_project_declarative_flow",
        lambda: "flow.yaml",
    )
    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.plot_declarative_flow_in_project_env",
        lambda definition: calls.append({"project_env": definition}),
    )

    plot_flow_module.plot_flow()

    assert calls == [{"project_env": "flow.yaml"}]


def test_plot_flow_keeps_python_entrypoint_without_definition(monkeypatch):
    subprocess_calls = []

    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.configured_project_declarative_flow",
        lambda: None,
    )
    monkeypatch.setattr(
        plot_flow_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    plot_flow_module.plot_flow()

    assert subprocess_calls == [
        (
            (["uv", "run", "plot"],),
            {"capture_output": False, "text": True, "check": True},
        )
    ]


def test_configured_project_declarative_flow(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.crewai]\ntype = "flow"\ndefinition = "flow.yaml"\n',
        encoding="utf-8",
    )

    from crewai_cli.run_declarative_flow import configured_project_declarative_flow

    assert configured_project_declarative_flow() == "flow.yaml"
