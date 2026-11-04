from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

import certifi
from crewai_cli.create_agent import create_agent
from crewai_cli.deploy.validate import DeployValidator
from crewai_core import plus_api
import pytest
import yaml


@pytest.fixture(params=["json", "yaml"])
def project(request, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    create_agent("support_agent", format=request.param)
    root = tmp_path / "support_agent"
    definition = root / f"agent.{request.param}"
    data = yaml.safe_load(definition.read_text())
    data.pop("tools")
    data.pop("skills")
    definition.write_text(json.dumps(data))
    (root / ".env").write_text("OPENAI_API_KEY=test-placeholder-key\n")
    return root


@pytest.fixture
def import_probe(project, monkeypatch, tmp_path):
    record = tmp_path / "transport.json"
    real_run = subprocess.run
    probe = (
        "\nfrom pathlib import Path\n"
        "from crewai_core import plus_api\n"
        f"Path({str(record)!r}).write_text(json.dumps({{"
        "'pinned': plus_api._MACHINE_TRANSPORT_ENV,"
        "'project_proxy': os.environ.get('HTTPS_PROXY'),"
        "'project_ca': os.environ.get('SSL_CERT_FILE')"
        "}))\n"
    )

    def run_in_test_environment(command, **kwargs):
        index = command.index("-c")
        return real_run(
            [sys.executable, "-c", command[index + 1] + probe, *command[index + 2 :]],
            **kwargs,
        )

    monkeypatch.setattr("crewai_cli.deploy.validate.subprocess.run", run_in_test_environment)
    monkeypatch.setattr("crewai_cli.deploy.validate.shutil.which", lambda name: name)
    return record


def test_agent_validation_pins_transport_before_project_dotenv(
    project, import_probe, monkeypatch
):
    for key in list(os.environ):
        if plus_api._sets_transport(key):
            monkeypatch.delenv(key)
    machine_ca = certifi.where()
    monkeypatch.setenv("HTTPS_PROXY", "http://machine-proxy.test:3128")
    monkeypatch.setenv("SSL_CERT_FILE", machine_ca)
    project_ca = project / "project-ca.pem"
    shutil.copyfile(machine_ca, project_ca)
    (project / ".env").write_text(
        "OPENAI_API_KEY=test-placeholder-key\n"
        "HTTPS_PROXY=http://project-proxy.test:8080\n"
        f"SSL_CERT_FILE={project_ca}\n"
    )

    validator = DeployValidator(project)
    assert validator._check_pyproject()
    validator._check_module_imports()

    assert validator.ok, validator.results
    observed = json.loads(import_probe.read_text())
    assert observed["pinned"] == {
        "HTTPS_PROXY": "http://machine-proxy.test:3128",
        "SSL_CERT_FILE": machine_ca,
    }
    assert observed["project_proxy"] == "http://project-proxy.test:8080"
    assert observed["project_ca"] == str(project_ca)


@pytest.mark.parametrize(
    "shadow", ["crewai_cli.py", "crewai_cli/__init__.py", "crewai_core/__init__.py"]
)
def test_agent_validation_does_not_import_project_cli_or_core(project, import_probe, shadow):
    marker = project / "shadow-executed"
    planted = project / shadow
    planted.parent.mkdir(parents=True, exist_ok=True)
    planted.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "raise RuntimeError('project shadow imported')\n"
    )

    validator = DeployValidator(project)
    assert validator._check_pyproject()
    validator._check_module_imports()

    assert not marker.exists()
    assert validator.ok, validator.results


@pytest.mark.parametrize("reference", ["custom", "import", "serialized", "callback", "agent_type"])
def test_declarative_agent_validation_does_not_execute_python(project, import_probe, reference):
    definition = next(project.glob("agent.*"))
    data = yaml.safe_load(definition.read_text())
    marker = project / "python-executed"
    malicious = (
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "raise RuntimeError('validation executed project Python')\n"
    )
    (project / "danger.py").write_text(malicious)
    tool = project / "tools" / "danger.py"
    tool.write_text(malicious)
    if reference == "custom":
        data["tools"] = ["custom:danger"]
    elif reference == "import":
        data["tools"] = ["danger:Tool"]
    elif reference == "serialized":
        data["tools"] = [{"tool_type": "danger.Tool"}]
    elif reference == "callback":
        data["step_callback"] = {"python": "danger.callback"}
    else:
        data["type"] = {"python": "danger.Agent"}
    definition.write_text(json.dumps(data))

    validator = DeployValidator(project)
    assert validator._check_pyproject()
    validator._check_module_imports()

    assert not marker.exists()
    assert validator.ok, validator.results


def test_declarative_agent_validation_reports_missing_tools(project, import_probe):
    definition = next(project.glob("agent.*"))
    data = yaml.safe_load(definition.read_text())
    data["tools"] = ["custom:missing"]
    definition.write_text(json.dumps(data))
    validator = DeployValidator(project)
    assert validator._check_pyproject()
    validator._check_module_imports()
    assert not validator.ok
    assert "custom:missing" in str(validator.results)
