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
    assert any(result.code == "invalid_agent_definition" for result in validator.errors)


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


@pytest.mark.parametrize("format", ["json", "yaml"])
def test_create_config_agent_loads_and_archives_without_source_package(
    format: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crewai.project import load_agent_project

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-placeholder-key")
    result = CliRunner().invoke(
        crewai, ["create", "agent", "Config-Agent", "--format", format]
    )
    assert result.exit_code == 0, result.output
    root = tmp_path / "config_agent"
    config = tomli.loads((root / "pyproject.toml").read_text())
    definition = f"agent.{format}"
    assert config["tool"]["crewai"]["definition"] == definition
    assert config["tool"]["hatch"]["build"]["targets"]["wheel"]["only-include"] == [
        definition,
        "tools",
    ]
    assert not (root / "src").exists()
    assert load_agent_project(root).role == "Support assistant"
    archive = create_project_zip("config_agent", project_dir=root)
    try:
        with zipfile.ZipFile(archive) as project_zip:
            assert definition in project_zip.namelist()
            assert not any(name.startswith("src/") for name in project_zip.namelist())
    finally:
        archive.unlink()


@pytest.mark.parametrize("format", ["json", "yaml"])
def test_config_agent_runs_same_loader_and_preserves_inputs(
    format: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    from crewai import Agent
    from crewai_cli.create_agent import create_agent
    from crewai_cli.run_crew import _run_agent

    monkeypatch.chdir(tmp_path)
    create_agent("config_agent", format=format)
    monkeypatch.chdir(tmp_path / "config_agent")
    (Path.cwd() / ".env").write_text("OPENAI_API_KEY=test-placeholder-key\n")
    messages = [{"role": "user", "content": "What are support hours?"}]
    received = []

    async def kickoff(self: Agent, **kwargs: Any) -> Any:
        from types import SimpleNamespace

        assert self.role == "Support assistant"
        received.append(kwargs["messages"])
        return SimpleNamespace(raw="finished")

    monkeypatch.setattr(Agent, "kickoff_async", kickoff)
    _run_agent(json.dumps({"messages": messages}))
    assert received == [messages]
    assert capsys.readouterr().out.endswith("finished\n")


@pytest.mark.parametrize("suffix", ["json", "yml", "yaml"])
def test_config_agent_validates_without_package_directory(
    suffix: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from crewai_cli.create_agent import create_agent

    monkeypatch.chdir(tmp_path)
    create_agent("config_agent", format="json")
    root = tmp_path / "config_agent"
    config_path = root / "pyproject.toml"
    config_path.write_text(
        config_path.read_text().replace("agent.json", f"agent.{suffix}")
    )
    definition = root / f"agent.{suffix}"
    if suffix != "json":
        (root / "agent.json").rename(definition)
        definition.write_text(
            "role: Support assistant\ngoal: Help users\nbackstory: A helpful support agent\nllm: openai/gpt-4o-mini\n"
        )
    else:
        assert json.loads(definition.read_text())["role"] == "Support assistant"
    (root / ".env").write_text("OPENAI_API_KEY=test-placeholder-key\n")
    (root / "uv.lock").write_text("# installed test environment")
    subprocess_run = subprocess.run

    def run_in_test_environment(
        command: list[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        if "-c" in command:
            return subprocess_run(
                [sys.executable, *command[command.index("-c") :]], **kwargs
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        "crewai_cli.deploy.validate.subprocess.run", run_in_test_environment
    )
    validator = DeployValidator(root)
    validator.run()
    assert validator.ok, validator.results
    assert not (root / "src").exists()


def test_create_format_option_rejects_non_agent_projects() -> None:
    result = CliRunner().invoke(
        crewai, ["create", "flow", "example", "--format", "json"]
    )
    assert result.exit_code == 2
    assert "--format can only be used with agent projects" in result.output


def test_config_agent_install_keeps_editable_project_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from crewai_cli.create_agent import create_agent
    from crewai_cli.install_crew import install_crew

    monkeypatch.chdir(tmp_path)
    create_agent("config_agent", format="json")
    monkeypatch.chdir(tmp_path / "config_agent")
    commands = []
    monkeypatch.setattr(
        "crewai_cli.install_crew.subprocess.run",
        lambda command, **kwargs: commands.append(command),
    )
    monkeypatch.setattr(
        "crewai_cli.install_crew.build_env_with_all_tool_credentials", dict
    )
    install_crew([])
    assert commands == [["uv", "sync"]]
