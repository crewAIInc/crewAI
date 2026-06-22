from __future__ import annotations

import json
import sys
import types

import pytest
import yaml

import crewai_cli.run_declarative_flow as run_declarative_flow_module
from crewai_cli.run_declarative_flow import run_declarative_flow


class _FakeFlow:
    def __init__(self, definition):
        self.definition = definition

    def kickoff(self, inputs=None):
        return {
            "flow": self.definition["name"],
            "inputs": inputs or {},
        }


class _FakeFlowFactory:
    @classmethod
    def from_definition(cls, definition):
        return _FakeFlow(definition)


class _FakeFlowDefinition:
    @classmethod
    def from_yaml(cls, source, *, source_path):
        return yaml.safe_load(source)

    @classmethod
    def from_json(cls, source, *, source_path):
        return json.loads(source)


@pytest.fixture
def fake_flow_runtime(monkeypatch):
    crewai_module = types.ModuleType("crewai")
    flow_package = types.ModuleType("crewai.flow")
    flow_module = types.ModuleType("crewai.flow.flow")
    flow_definition_module = types.ModuleType("crewai.flow.flow_definition")

    flow_module.Flow = _FakeFlowFactory
    flow_definition_module.FlowDefinition = _FakeFlowDefinition

    monkeypatch.setitem(sys.modules, "crewai", crewai_module)
    monkeypatch.setitem(sys.modules, "crewai.flow", flow_package)
    monkeypatch.setitem(sys.modules, "crewai.flow.flow", flow_module)
    monkeypatch.setitem(
        sys.modules, "crewai.flow.flow_definition", flow_definition_module
    )


def _captured_json(capsys):
    return json.loads(capsys.readouterr().out)


def test_run_declarative_flow_reads_definition_file(
    tmp_path, capsys, fake_flow_runtime
):
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text("schema: crewai.flow/v1\nname: TestFlow\n")

    run_declarative_flow(str(definition_path), '{"topic":"AI"}')

    assert _captured_json(capsys) == {
        "flow": "TestFlow",
        "inputs": {"topic": "AI"},
    }


@pytest.mark.parametrize(
    ("filename", "definition_source", "expected_flow_name"),
    [
        pytest.param(
            "flow.yaml",
            "schema: crewai.flow/v1\nname: YamlFileFlow\n",
            "YamlFileFlow",
            id="yaml-file",
        ),
        pytest.param(
            "flow.json",
            '{"schema":"crewai.flow/v1","name":"JsonFlow"}',
            "JsonFlow",
            id="json-file",
        ),
    ],
)
def test_run_declarative_flow_accepts_definition_files(
    filename, definition_source, expected_flow_name, tmp_path, capsys, fake_flow_runtime
):
    definition_path = tmp_path / filename
    definition_path.write_text(definition_source)

    run_declarative_flow(str(definition_path))

    assert _captured_json(capsys) == {"flow": expected_flow_name, "inputs": {}}


def test_run_declarative_flow_rejects_non_object_inputs(
    tmp_path, fake_flow_runtime, capsys
):
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text("schema: crewai.flow/v1\nname: TestFlow\n")

    with pytest.raises(SystemExit):
        run_declarative_flow(str(definition_path), '["not", "an", "object"]')

    assert "Invalid --inputs JSON: expected an object." in capsys.readouterr().err


def test_run_declarative_flow_reports_unreadable_file(
    monkeypatch, tmp_path, capsys, fake_flow_runtime
):
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text("schema: crewai.flow/v1\nname: TestFlow\n")

    def raise_permission_error(self, *args, **kwargs):
        raise PermissionError("no access")

    monkeypatch.setattr("pathlib.Path.read_text", raise_permission_error)

    with pytest.raises(SystemExit):
        run_declarative_flow(str(definition_path))

    err = capsys.readouterr().err
    assert "Unable to read --definition path" in err
    assert str(definition_path) in err
    assert "no access" in err


def test_run_declarative_flow_reports_missing_file(capsys, fake_flow_runtime):
    with pytest.raises(SystemExit):
        run_declarative_flow("missing-flow.yaml")

    assert (
        "Invalid --definition path: missing-flow.yaml does not exist."
        in capsys.readouterr().err
    )


def test_run_declarative_flow_in_project_env_uses_uv(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    subprocess_calls = []

    monkeypatch.setattr(
        run_declarative_flow_module,
        "build_env_with_all_tool_credentials",
        lambda: {"EXISTING": "value"},
    )

    def fake_subprocess_run(command, **kwargs):
        subprocess_calls.append((command, kwargs))

    monkeypatch.setattr(
        run_declarative_flow_module.subprocess, "run", fake_subprocess_run
    )

    run_declarative_flow_module.run_declarative_flow_in_project_env(
        "flow.yaml", '{"topic":"AI"}'
    )

    assert subprocess_calls == [
        (
            [
                "uv",
                "run",
                "crewai",
                "run",
                "--definition",
                "flow.yaml",
                "--inputs",
                '{"topic":"AI"}',
            ],
            {
                "capture_output": False,
                "text": True,
                "check": True,
                "env": {"EXISTING": "value"},
            },
        )
    ]


def test_run_declarative_flow_in_project_env_falls_back_without_pyproject(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr(
        run_declarative_flow_module,
        "run_declarative_flow",
        lambda **kwargs: calls.append(kwargs),
    )

    run_declarative_flow_module.run_declarative_flow_in_project_env("flow.yaml")

    assert calls == [{"definition": "flow.yaml", "inputs": None}]


def test_run_declarative_flow_in_project_env_uses_in_process_runner_inside_uv(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    calls = []
    subprocess_calls = []

    monkeypatch.setattr(
        run_declarative_flow_module,
        "run_declarative_flow",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        run_declarative_flow_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    run_declarative_flow_module.run_declarative_flow_in_project_env(
        "flow.yaml", '{"topic":"AI"}'
    )

    assert calls == [{"definition": "flow.yaml", "inputs": '{"topic":"AI"}'}]
    assert subprocess_calls == []


def test_plot_declarative_flow_in_project_env_uses_uv(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("UV_RUN_RECURSION_DEPTH", raising=False)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    subprocess_calls = []

    monkeypatch.setattr(
        run_declarative_flow_module,
        "build_env_with_all_tool_credentials",
        lambda: {},
    )

    def fake_subprocess_run(command, **kwargs):
        subprocess_calls.append((command, kwargs))

    monkeypatch.setattr(
        run_declarative_flow_module.subprocess, "run", fake_subprocess_run
    )

    run_declarative_flow_module.plot_declarative_flow_in_project_env("flow.yaml")

    assert subprocess_calls == [
        (
            [
                "uv",
                "run",
                "crewai",
                "flow",
                "plot",
            ],
            {
                "capture_output": False,
                "text": True,
                "check": True,
                "env": {},
            },
        )
    ]


def test_plot_declarative_flow_in_project_env_uses_in_process_runner_inside_uv(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    calls = []
    subprocess_calls = []

    monkeypatch.setattr(
        run_declarative_flow_module,
        "plot_declarative_flow",
        lambda **kwargs: calls.append(kwargs),
    )
    monkeypatch.setattr(
        run_declarative_flow_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess_calls.append((args, kwargs)),
    )

    run_declarative_flow_module.plot_declarative_flow_in_project_env("flow.yaml")

    assert calls == [{"definition": "flow.yaml"}]
    assert subprocess_calls == []
