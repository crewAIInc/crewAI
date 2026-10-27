from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile

from click.testing import CliRunner
from crewai_cli.cli import crewai
from crewai_cli.deploy.archive import create_project_zip
from crewai_cli.deploy.validate import DeployValidator
import pytest
import tomli


@pytest.fixture
def agent_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(crewai, ["create", "agent", "Support-Agent"])
    assert result.exit_code == 0, result.output
    return tmp_path / "support_agent"


def test_create_agent_exports_installable_agent(
    agent_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crewai import Agent
    from crewai.project import load_agent_project

    config = tomli.loads((agent_project / "pyproject.toml").read_text())
    assert config["project"]["name"] == "support_agent"
    assert config["tool"]["crewai"]["type"] == "agent"
    assert config["tool"]["crewai"]["project_id"]
    assert config["build-system"]["build-backend"] == "hatchling.build"
    assert config["project"]["dependencies"]
    assert "scripts" not in config["project"]
    package = agent_project / "src" / "support_agent"
    assert not (package / "main.py").exists()
    assert not (package / "crew.py").exists()
    assert not (package / "config").exists()
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-key")
    agent = load_agent_project(agent_project)
    assert isinstance(agent, Agent)
    assert (
        agent.tools[0].run()
        == "Support is available Monday to Friday, 09:00-17:00 UTC."
    )


def test_agent_archive_includes_source_and_excludes_secrets(
    agent_project: Path,
) -> None:
    (agent_project / ".env").write_text("OPENAI_API_KEY=private-test-value")
    archive = create_project_zip("support_agent", project_dir=agent_project)
    try:
        with zipfile.ZipFile(archive) as project_zip:
            names = project_zip.namelist()
            assert "src/support_agent/agent.py" in names
            assert "pyproject.toml" in names
            assert ".env.example" in names
            assert ".env" not in names
            assert not any(name.startswith(".git/") for name in names)
    finally:
        archive.unlink()


@pytest.mark.parametrize("name", ["123-agent", "class", "../", "!!!"])
def test_create_agent_rejects_invalid_package_name(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(crewai, ["create", "agent", "--", name])
    assert result.exit_code != 0
    assert "Python package" in result.output
    assert not list(tmp_path.iterdir())


def test_create_agent_preserves_existing_directory(agent_project: Path) -> None:
    source = agent_project / "src" / "support_agent" / "agent.py"
    source.write_text("original")
    result = CliRunner().invoke(crewai, ["create", "agent", "Support-Agent"])
    assert result.exit_code != 0
    assert "already exists" in result.output
    assert source.read_text() == "original"


@pytest.mark.parametrize(
    "flag",
    [["--classic"], ["--declarative"], ["--provider", "openai"], ["--skip-provider"]],
)
def test_create_agent_rejects_crew_options(flag: list[str]) -> None:
    result = CliRunner().invoke(crewai, ["create", "agent", "example", *flag])
    assert result.exit_code == 2
    assert "cannot be used with agent projects" in result.output


@pytest.mark.parametrize(
    "source", [None, "agent = object()", "other = object()", "syntax error!"]
)
def test_agent_deploy_validation_checks_real_export(
    agent_project: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str | None,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-key")
    (agent_project / "uv.lock").write_text("# installed test environment")
    if source is not None:
        (agent_project / "src" / "support_agent" / "agent.py").write_text(source)
    subprocess_run = subprocess.run

    def run_in_test_environment(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if "-c" in command:
            assert "--no-sync" in command
            return subprocess_run(
                [sys.executable, *command[command.index("-c") :]], **kwargs
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "crewai_cli.deploy.validate.subprocess.run", run_in_test_environment
    )
    validator = DeployValidator(agent_project)
    validator.run()
    if source is None:
        assert validator.ok, validator.results
    else:
        assert not validator.ok
        assert any(result.code == "import_failed" for result in validator.errors)
    assert not any(
        result.code in {"missing_crew_py", "missing_config_dir", "missing_flow_main"}
        for result in validator.results
    )


def test_agent_deploy_validation_requires_agent_module(
    agent_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (agent_project / "src" / "support_agent" / "agent.py").unlink()
    monkeypatch.setattr(DeployValidator, "_check_module_imports", lambda self: None)
    validator = DeployValidator(agent_project)
    validator.run()
    assert any(result.code == "missing_agent_py" for result in validator.errors)


def test_agent_deploy_validation_loads_project_dotenv_in_subprocess(
    agent_project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MANAGED_AGENT_VALIDATION_ENV", raising=False)
    (agent_project / ".env").write_text(
        "OPENAI_API_KEY=project-placeholder-key\n"
        "MODEL=openai/gpt-4o-mini\n"
        "MANAGED_AGENT_VALIDATION_ENV=from-project-dotenv\n"
    )
    source = agent_project / "src" / "support_agent" / "agent.py"
    source.write_text(
        "import os\n"
        'assert os.environ.get("OPENAI_API_KEY") == "project-placeholder-key"\n'
        'assert os.environ.get("MANAGED_AGENT_VALIDATION_ENV") == "from-project-dotenv"\n'
        + source.read_text()
    )
    subprocess_run = subprocess.run

    def run_in_test_environment(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        return subprocess_run(
            [sys.executable, *command[command.index("-c") :]], **kwargs
        )

    monkeypatch.setattr(
        "crewai_cli.deploy.validate.subprocess.run", run_in_test_environment
    )
    validator = DeployValidator(agent_project)
    assert validator._check_pyproject()
    validator._check_module_imports()
    assert validator.ok, validator.results
    assert "OPENAI_API_KEY" not in os.environ
    assert "MANAGED_AGENT_VALIDATION_ENV" not in os.environ
