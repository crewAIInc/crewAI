"""Tests for crewai_cli.run_crew JSON crew handling."""

import os
from pathlib import Path
import subprocess
import sys

import pytest
from crewai_core.constants import CREWAI_TRAINED_AGENTS_FILE_ENV

import crewai_cli.run_crew as run_crew_module


def test_run_crew_forwards_trained_agents_file_to_json_crews(monkeypatch):
    """crewai run -f must reach JSON crews, not only classic subprocess crews."""
    monkeypatch.setattr(run_crew_module, "_has_json_crew", lambda: True)
    called: dict = {}

    def fake_run_json_crew_in_project_env(trained_agents_file=None):
        called["trained_agents_file"] = trained_agents_file

    monkeypatch.setattr(
        run_crew_module,
        "_run_json_crew_in_project_env",
        fake_run_json_crew_in_project_env,
    )

    run_crew_module.run_crew(trained_agents_file="some.pkl")

    assert called == {"trained_agents_file": "some.pkl"}


def test_json_run_uses_project_env_when_pyproject_exists(monkeypatch, tmp_path: Path):
    """JSON crew runs should execute inside the project uv environment."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    install_calls = []
    subprocess_calls = []

    monkeypatch.setattr(
        run_crew_module,
        "_install_json_crew_dependencies_if_needed",
        lambda: install_calls.append(True),
    )
    monkeypatch.setattr(
        run_crew_module,
        "build_env_with_all_tool_credentials",
        lambda: {"EXISTING": "value"},
    )

    def fake_subprocess_run(command, **kwargs):
        subprocess_calls.append((command, kwargs))

    monkeypatch.setattr(run_crew_module.subprocess, "run", fake_subprocess_run)

    run_crew_module._run_json_crew_in_project_env(
        trained_agents_file="trained.pkl"
    )

    expected_env = {
        "EXISTING": "value",
        run_crew_module._CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV: str(
            Path(run_crew_module.__file__).resolve().parent
        ),
        CREWAI_TRAINED_AGENTS_FILE_ENV: "trained.pkl",
    }
    if local_crewai_source_dir := run_crew_module._find_local_crewai_source_dir():
        expected_env[run_crew_module._CREWAI_RUNNER_SOURCE_DIR_ENV] = str(
            local_crewai_source_dir
        )

    assert install_calls == [True]
    assert subprocess_calls == [
        (
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-c",
                run_crew_module._JSON_CREW_RUNNER_CODE,
            ],
            {
                "capture_output": False,
                "text": True,
                "check": True,
                "env": expected_env,
            },
        )
    ]


def test_json_runner_code_loads_current_cli_package_over_project_env(tmp_path: Path):
    old_parent = tmp_path / "old"
    old_pkg = old_parent / "crewai_cli"
    old_pkg.mkdir(parents=True)
    (old_pkg / "__init__.py").write_text("")
    (old_pkg / "run_crew.py").write_text("raise ImportError('old package used')\n")
    old_crewai_project = old_parent / "crewai" / "project"
    old_crewai_project.mkdir(parents=True)
    (old_parent / "crewai" / "__init__.py").write_text("")
    (old_crewai_project / "__init__.py").write_text("")
    (old_crewai_project / "json_loader.py").write_text(
        "raise ImportError('old crewai used')\n"
    )

    current_pkg = tmp_path / "current" / "crewai_cli"
    current_pkg.mkdir(parents=True)
    marker = tmp_path / "marker.txt"
    (current_pkg / "__init__.py").write_text("")
    (current_pkg / "run_crew.py").write_text(
        "from pathlib import Path\n"
        "from crewai.project.json_loader import SOURCE\n"
        "def _run_json_crew(trained_agents_file=None):\n"
        f"    Path({str(marker)!r}).write_text(SOURCE + ':' + (trained_agents_file or ''))\n"
    )
    current_crewai_project = tmp_path / "current_crewai_src" / "crewai" / "project"
    current_crewai_project.mkdir(parents=True)
    (tmp_path / "current_crewai_src" / "crewai" / "__init__.py").write_text("")
    (current_crewai_project / "__init__.py").write_text("")
    (current_crewai_project / "json_loader.py").write_text("SOURCE = 'current'\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(old_parent)
    env[run_crew_module._CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV] = str(current_pkg)
    env[run_crew_module._CREWAI_RUNNER_SOURCE_DIR_ENV] = str(
        tmp_path / "current_crewai_src"
    )
    env[CREWAI_TRAINED_AGENTS_FILE_ENV] = "trained.pkl"

    subprocess.run(
        [sys.executable, "-c", run_crew_module._JSON_CREW_RUNNER_CODE],
        check=True,
        env=env,
        cwd=tmp_path,
    )

    assert marker.read_text() == "current:trained.pkl"


def test_json_run_without_pyproject_runs_in_process(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    called: dict = {}

    def fake_run_json_crew(trained_agents_file=None):
        called["trained_agents_file"] = trained_agents_file
        return "result"

    monkeypatch.setattr(run_crew_module, "_run_json_crew", fake_run_json_crew)

    assert (
        run_crew_module._run_json_crew_in_project_env(
            trained_agents_file="trained.pkl"
        )
        == "result"
    )
    assert called == {"trained_agents_file": "trained.pkl"}


def test_json_project_env_run_failure_exits_nonzero(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")

    monkeypatch.setattr(
        run_crew_module, "_install_json_crew_dependencies_if_needed", lambda: None
    )
    monkeypatch.setattr(
        run_crew_module, "build_env_with_all_tool_credentials", lambda: {}
    )

    def fake_subprocess_run(command, **kwargs):
        raise subprocess.CalledProcessError(7, command)

    monkeypatch.setattr(run_crew_module.subprocess, "run", fake_subprocess_run)

    with pytest.raises(SystemExit) as exc_info:
        run_crew_module._run_json_crew_in_project_env()

    assert exc_info.value.code == 7


def test_json_run_installs_dependencies_when_pyproject_has_no_lockfile(
    monkeypatch, tmp_path: Path
):
    """JSON crew runs should lock/sync project dependencies only once."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    calls = []

    def fake_install_crew(
        proxy_options, *, raise_on_error=False, install_project=None
    ):
        calls.append((proxy_options, raise_on_error, install_project))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    run_crew_module._install_json_crew_dependencies_if_needed()

    assert calls == [([], True, False)]


@pytest.mark.parametrize("lockfile", ["uv.lock", "poetry.lock"])
def test_json_run_skips_dependency_install_when_lockfile_exists(
    monkeypatch, tmp_path: Path, lockfile: str
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / lockfile).write_text("# lock\n")
    calls = []

    def fake_install_crew(
        proxy_options, *, raise_on_error=False, install_project=None
    ):
        calls.append((proxy_options, raise_on_error, install_project))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    run_crew_module._install_json_crew_dependencies_if_needed()

    assert calls == []


def test_json_run_skips_dependency_install_without_pyproject(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_install_crew(
        proxy_options, *, raise_on_error=False, install_project=None
    ):
        calls.append((proxy_options, raise_on_error))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    run_crew_module._install_json_crew_dependencies_if_needed()

    assert calls == []


def test_json_run_install_failure_exits_nonzero(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")

    def fake_install_crew(
        proxy_options, *, raise_on_error=False, install_project=None
    ):
        raise subprocess.CalledProcessError(42, ["uv", "sync"])

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    with pytest.raises(SystemExit) as exc_info:
        run_crew_module._install_json_crew_dependencies_if_needed()

    assert exc_info.value.code == 42


def test_run_json_crew_exports_trained_agents_env(monkeypatch, tmp_path: Path):
    """JSON crews run in-process, so the pickle path must land in the env var."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CREWAI_TRAINED_AGENTS_FILE_ENV, raising=False)

    try:
        # No crew.json(c) in tmp_path: the loader fails *after* the env var
        # export, which is the part under test.
        with pytest.raises(FileNotFoundError):
            run_crew_module._run_json_crew(trained_agents_file="some.pkl")
        assert os.environ[CREWAI_TRAINED_AGENTS_FILE_ENV] == "some.pkl"
    finally:
        os.environ.pop(CREWAI_TRAINED_AGENTS_FILE_ENV, None)


def test_run_json_crew_leaves_env_untouched_without_flag(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CREWAI_TRAINED_AGENTS_FILE_ENV, raising=False)

    with pytest.raises(FileNotFoundError):
        run_crew_module._run_json_crew()

    assert CREWAI_TRAINED_AGENTS_FILE_ENV not in os.environ


def test_missing_input_names_accepts_hyphenated_placeholders():
    """The prompt regex must accept the same names kickoff interpolation does."""
    from types import SimpleNamespace

    crew = SimpleNamespace(
        agents=[
            SimpleNamespace(
                role="Researcher", goal="Cover {my-topic}", backstory=""
            )
        ],
        tasks=[
            SimpleNamespace(
                description="Write about {my-topic} for {target-audience}",
                expected_output="Post",
                output_file=None,
            )
        ],
    )

    assert run_crew_module._missing_input_names(crew, {}) == [
        "my-topic",
        "target-audience",
    ]


def _patch_tui_run(monkeypatch, status: str):
    """Stub the TUI pieces of _run_json_crew so only exit handling runs."""

    class FakeApp:
        def __init__(self, **kwargs):
            self._status = status
            self._crew_result = "result" if status == "completed" else None
            self._want_deploy = False

        def run(self):
            pass

    from types import SimpleNamespace

    crew = SimpleNamespace(name="Demo", tasks=[], agents=[])
    monkeypatch.setattr(
        run_crew_module, "find_crew_json_file", lambda: Path("crew.jsonc")
    )
    monkeypatch.setattr(
        run_crew_module,
        "_load_json_crew_for_tui",
        lambda _path: (FakeApp, crew, {}, [], []),
    )
    monkeypatch.setattr(
        run_crew_module, "_prompt_for_missing_inputs", lambda _crew, inputs: inputs
    )
    monkeypatch.setattr(run_crew_module, "_print_post_tui_summary", lambda _app: None)


def test_run_json_crew_failed_status_exits_nonzero(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _patch_tui_run(monkeypatch, status="failed")

    with pytest.raises(SystemExit) as exc_info:
        run_crew_module._run_json_crew()

    assert exc_info.value.code == 1


def test_run_json_crew_completed_status_returns_result(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    _patch_tui_run(monkeypatch, status="completed")

    assert run_crew_module._run_json_crew() == "result"


def test_has_json_crew_defers_to_declared_flow_type(monkeypatch, tmp_path: Path):
    """A flow project containing a stray crew.jsonc must still run as a flow."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "crew.jsonc").write_text("{}")
    (tmp_path / "pyproject.toml").write_text('[tool.crewai]\ntype = "flow"\n')

    assert run_crew_module._has_json_crew() is False


def test_has_json_crew_true_for_declared_crew_type(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "crew.jsonc").write_text("{}")
    (tmp_path / "pyproject.toml").write_text('[tool.crewai]\ntype = "crew"\n')

    assert run_crew_module._has_json_crew() is True


def test_has_json_crew_true_without_pyproject(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "crew.jsonc").write_text("{}")

    assert run_crew_module._has_json_crew() is True
