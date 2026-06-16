from __future__ import annotations

import json
import sys
import types

import pytest
import yaml

from crewai_cli.run_flow_definition import run_flow_definition


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
    def from_yaml(cls, source):
        return yaml.safe_load(source)

    @classmethod
    def from_json(cls, source):
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


def test_run_flow_definition_reads_definition_file(
    tmp_path, capsys, fake_flow_runtime
):
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text("schema: crewai.flow/v1\nname: TestFlow\n")

    run_flow_definition(str(definition_path), '{"topic":"AI"}')

    assert _captured_json(capsys) == {
        "flow": "TestFlow",
        "inputs": {"topic": "AI"},
    }


@pytest.mark.parametrize(
    ("definition_source", "expected_flow_name"),
    [
        pytest.param(
            "schema: crewai.flow/v1\nname: InlineFlow\n",
            "InlineFlow",
            id="inline-yaml",
        ),
        pytest.param(
            '{"schema":"crewai.flow/v1","name":"InlineJsonFlow"}',
            "InlineJsonFlow",
            id="inline-json",
        ),
        pytest.param(
            '{"schema":"crewai.flow/v1","name":"' + ("JsonFlow" * 500) + '"}',
            "JsonFlow" * 500,
            id="large-inline-json",
        ),
    ],
)
def test_run_flow_definition_accepts_inline_definitions(
    definition_source, expected_flow_name, capsys, fake_flow_runtime
):
    run_flow_definition(definition_source)

    assert _captured_json(capsys) == {"flow": expected_flow_name, "inputs": {}}


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
def test_run_flow_definition_accepts_definition_files(
    filename, definition_source, expected_flow_name, tmp_path, capsys, fake_flow_runtime
):
    definition_path = tmp_path / filename
    definition_path.write_text(definition_source)

    run_flow_definition(str(definition_path))

    assert _captured_json(capsys) == {"flow": expected_flow_name, "inputs": {}}


def test_run_flow_definition_makes_definition_dir_importable(
    tmp_path, capsys, fake_flow_runtime, monkeypatch
):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / "lead_flow.py").write_text("MARKER = 'loaded'\n")
    definition_path = project_dir / "flow.yaml"
    definition_path.write_text("schema: crewai.flow/v1\nname: TestFlow\n")

    monkeypatch.delitem(sys.modules, "lead_flow", raising=False)
    monkeypatch.chdir(tmp_path)

    run_flow_definition(str(definition_path))

    import lead_flow

    assert lead_flow.MARKER == "loaded"


def test_run_flow_definition_rejects_non_object_inputs(fake_flow_runtime, capsys):
    with pytest.raises(SystemExit):
        run_flow_definition("name: TestFlow", '["not", "an", "object"]')

    assert "Invalid --inputs JSON: expected an object." in capsys.readouterr().err


def test_run_flow_definition_reports_unreadable_file(
    monkeypatch, tmp_path, capsys, fake_flow_runtime
):
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text("schema: crewai.flow/v1\nname: TestFlow\n")

    def raise_permission_error(self, *args, **kwargs):
        raise PermissionError("no access")

    monkeypatch.setattr("pathlib.Path.read_text", raise_permission_error)

    with pytest.raises(SystemExit):
        run_flow_definition(str(definition_path))

    err = capsys.readouterr().err
    assert "Unable to read --definition path" in err
    assert str(definition_path) in err
    assert "no access" in err
