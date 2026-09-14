from pathlib import Path
import subprocess

import pytest

import crewai_cli.install_crew as install_crew_module


@pytest.fixture(autouse=True)
def _tool_credentials(monkeypatch):
    monkeypatch.setattr(
        install_crew_module,
        "build_env_with_all_tool_credentials",
        lambda: {"CREWAI_TEST": "1"},
    )


def test_install_crew_json_project_skips_project_install(
    fp, monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "json_crew"

[tool.crewai]
type = "crew"
definition = "crew.jsonc"
""".strip()
    )
    (tmp_path / "crew.jsonc").write_text("{}\n")
    fp.register(["uv", "sync", "--no-install-project"], stdout="")

    install_crew_module.install_crew([])


def test_install_crew_json_project_with_python_package_installs_project(
    fp, monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "hybrid-crew"

[tool.crewai]
type = "crew"
definition = "crew.jsonc"
""".strip()
    )
    (tmp_path / "crew.jsonc").write_text("{}\n")
    package_dir = tmp_path / "src" / "hybrid_crew"
    package_dir.mkdir(parents=True)
    (package_dir / "crew.py").write_text("class HybridCrew: ...\n")
    fp.register(["uv", "sync"], stdout="")

    install_crew_module.install_crew([])


def test_install_crew_flow_project_installs_project(fp, monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "flow_project"

[tool.crewai]
type = "flow"
""".strip()
    )
    (tmp_path / "crew.jsonc").write_text("{}\n")
    fp.register(["uv", "sync"], stdout="")

    install_crew_module.install_crew([])


def test_install_crew_classic_project_installs_project(
    fp, monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'classic'\n")
    fp.register(["uv", "sync"], stdout="")

    install_crew_module.install_crew([])


def test_install_crew_install_project_false_adds_no_install_project(fp):
    fp.register(["uv", "sync", "--no-install-project", "--frozen"], stdout="")

    install_crew_module.install_crew(["--frozen"], install_project=False)


def test_install_crew_reraises_sync_failure_when_requested(fp):
    fp.register(["uv", "sync"], returncode=1, stderr="sync failed\n")

    with pytest.raises(subprocess.CalledProcessError):
        install_crew_module.install_crew([], raise_on_error=True)


def test_install_crew_swallows_sync_failure_by_default(fp):
    fp.register(["uv", "sync"], returncode=1, stderr="sync failed\n")

    install_crew_module.install_crew([])
