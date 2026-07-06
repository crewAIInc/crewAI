"""Tests for crewai_cli.run_crew JSON crew handling."""

import os
from pathlib import Path
import subprocess
import sys

import click
import pytest
from crewai_core.constants import CREWAI_TRAINED_AGENTS_FILE_ENV

import crewai_cli.run_crew as run_crew_module


def test_missing_crewai_package_shows_full_install_hint(monkeypatch):
    def missing_crewai_package():
        raise ModuleNotFoundError("No module named 'crewai'", name="crewai")

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "crewai.project.crew_loader":
            missing_crewai_package()
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(click.ClickException) as exc_info:
        run_crew_module._load_json_crew(Path("crew.jsonc"))

    message = exc_info.value.message
    assert "CrewAI CLI is installed without the `crewai` package" in message
    assert "uv tool install --force 'crewai[tools]>=1.15.0,<2.0.0'" in message
    assert "quotes are required in zsh" in message


def test_run_crew_forwards_trained_agents_file_to_json_crews(monkeypatch):
    """crewai run -f must reach JSON crews, not only classic subprocess crews."""
    monkeypatch.setattr(run_crew_module, "read_toml", lambda: {})
    monkeypatch.setattr(
        run_crew_module,
        "configured_project_json_crew",
        lambda pyproject_data=None, project_root=None: Path("crew.jsonc"),
    )
    called: dict = {}

    def fake_run_json_crew_in_project_env(trained_agents_file=None, crew_path=None):
        called["trained_agents_file"] = trained_agents_file
        called["crew_path"] = crew_path

    monkeypatch.setattr(
        run_crew_module,
        "_run_json_crew_in_project_env",
        fake_run_json_crew_in_project_env,
    )

    run_crew_module.run_crew(trained_agents_file="some.pkl")

    assert called == {
        "trained_agents_file": "some.pkl",
        "crew_path": Path("crew.jsonc"),
    }


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

    crew_path = tmp_path / "crew.jsonc"
    run_crew_module._run_json_crew_in_project_env(
        trained_agents_file="trained.pkl",
        crew_path=crew_path,
    )

    expected_env = {
        "EXISTING": "value",
        run_crew_module._CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV: str(
            Path(run_crew_module.__file__).resolve().parent
        ),
        CREWAI_TRAINED_AGENTS_FILE_ENV: "trained.pkl",
        run_crew_module._CREWAI_JSON_CREW_DEFINITION_ENV: str(crew_path),
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


def test_json_run_uses_poetry_run_for_poetry_lock_without_uv_lock(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "poetry.lock").write_text("# lock\n")
    monkeypatch.setattr(
        run_crew_module,
        "_install_json_crew_dependencies_if_needed",
        lambda: None,
    )
    monkeypatch.setattr(
        run_crew_module,
        "build_env_with_all_tool_credentials",
        lambda: {},
    )
    subprocess_calls = []

    def fake_subprocess_run(command, **kwargs):
        subprocess_calls.append((command, kwargs))

    monkeypatch.setattr(run_crew_module.subprocess, "run", fake_subprocess_run)

    run_crew_module._run_json_crew_in_project_env()

    expected_env = {
        run_crew_module._CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV: str(
            Path(run_crew_module.__file__).resolve().parent
        ),
    }
    if local_crewai_source_dir := run_crew_module._find_local_crewai_source_dir():
        expected_env[run_crew_module._CREWAI_RUNNER_SOURCE_DIR_ENV] = str(
            local_crewai_source_dir
        )

    assert subprocess_calls == [
        (
            [
                "poetry",
                "run",
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


def test_json_runner_imports_with_older_project_env_crewai_core(tmp_path: Path):
    old_parent = tmp_path / "old_env"
    old_crewai_core = old_parent / "crewai_core"
    old_crewai_core.mkdir(parents=True)
    (old_crewai_core / "__init__.py").write_text("")
    (old_crewai_core / "constants.py").write_text(
        "CREWAI_TRAINED_AGENTS_FILE_ENV = 'CREWAI_TRAINED_AGENTS_FILE'\n"
    )
    (old_crewai_core / "project.py").write_text(
        "def read_toml(*args, **kwargs):\n"
        "    return {}\n"
        "def parse_toml(*args, **kwargs):\n"
        "    return {}\n"
        "def get_project_description(*args, **kwargs):\n"
        "    return None\n"
        "def get_project_name(*args, **kwargs):\n"
        "    return None\n"
        "def get_project_version(*args, **kwargs):\n"
        "    return None\n"
    )
    (old_crewai_core / "tool_credentials.py").write_text(
        "def build_env_with_all_tool_credentials(*args, **kwargs):\n"
        "    return {}\n"
        "def build_env_with_tool_repository_credentials(*args, **kwargs):\n"
        "    return {}\n"
    )
    (old_crewai_core / "version.py").write_text(
        "def check_version(*args, **kwargs):\n"
        "    return None\n"
        "def get_crewai_version(*args, **kwargs):\n"
        "    return '1.0.0'\n"
        "def get_latest_version_from_pypi(*args, **kwargs):\n"
        "    return None\n"
        "def is_current_version_yanked(*args, **kwargs):\n"
        "    return False\n"
        "def is_newer_version_available(*args, **kwargs):\n"
        "    return False\n"
    )

    marker = tmp_path / "marker.txt"
    old_crewai_project = old_parent / "crewai" / "project"
    old_crewai_project.mkdir(parents=True)
    (old_parent / "crewai" / "__init__.py").write_text("")
    (old_crewai_project / "__init__.py").write_text("")
    (old_crewai_project / "crew_loader.py").write_text(
        "from pathlib import Path\n"
        "class Crew:\n"
        "    agents = []\n"
        "    tasks = []\n"
        "    def kickoff(self, inputs):\n"
        f"        Path({str(marker)!r}).write_text('ran')\n"
        "        return 'done'\n"
        "def load_crew(path):\n"
        "    return Crew(), {}\n"
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = str(old_parent)
    env["CREWAI_DMN"] = "true"
    env[run_crew_module._CREWAI_CLI_RUNNER_PACKAGE_DIR_ENV] = str(
        Path(run_crew_module.__file__).resolve().parent
    )
    env[run_crew_module._CREWAI_JSON_CREW_DEFINITION_ENV] = "crew.jsonc"

    subprocess.run(
        [sys.executable, "-c", run_crew_module._JSON_CREW_RUNNER_CODE],
        check=True,
        env=env,
        cwd=tmp_path,
    )

    assert marker.read_text() == "ran"


def test_json_run_without_pyproject_runs_in_process(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    called: dict = {}

    def fake_run_json_crew(trained_agents_file=None, crew_path=None):
        called["trained_agents_file"] = trained_agents_file
        called["crew_path"] = crew_path
        return "result"

    monkeypatch.setattr(run_crew_module, "_run_json_crew", fake_run_json_crew)

    assert (
        run_crew_module._run_json_crew_in_project_env(
            trained_agents_file="trained.pkl"
        )
        == "result"
    )
    assert called == {"trained_agents_file": "trained.pkl", "crew_path": None}


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
    """JSON crew runs should lock/sync project dependencies only when needed."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    calls = []

    def fake_install_crew(
        proxy_options, *, raise_on_error=False, install_project=None
    ):
        calls.append((proxy_options, raise_on_error, install_project))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    run_crew_module._install_json_crew_dependencies_if_needed()

    assert calls == [([], True, None)]


def test_json_run_syncs_frozen_when_uv_lock_exists_without_venv(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "uv.lock").write_text("# lock\n")
    calls = []

    def fake_install_crew(
        proxy_options, *, raise_on_error=False, install_project=None
    ):
        calls.append((proxy_options, raise_on_error, install_project))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    run_crew_module._install_json_crew_dependencies_if_needed()

    assert calls == [(["--frozen"], True, None)]


def test_json_run_skips_uv_sync_when_only_poetry_lock_exists_without_venv(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "poetry.lock").write_text("# lock\n")
    calls = []

    def fake_install_crew(
        proxy_options, *, raise_on_error=False, install_project=None
    ):
        calls.append((proxy_options, raise_on_error, install_project))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    run_crew_module._install_json_crew_dependencies_if_needed()

    assert calls == []


@pytest.mark.parametrize("lockfile", ["uv.lock", "poetry.lock"])
def test_json_run_skips_dependency_install_when_lockfile_and_venv_exist(
    monkeypatch, tmp_path: Path, lockfile: str
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / lockfile).write_text("# lock\n")
    (tmp_path / ".venv").mkdir()
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
    monkeypatch.delenv("CREWAI_DMN", raising=False)

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
        run_crew_module, "configured_project_json_crew", lambda: Path("crew.jsonc")
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


def test_run_json_crew_dmn_mode_bypasses_tui(monkeypatch, tmp_path: Path, capsys):
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")
    crew_path = tmp_path / "crew.jsonc"
    crew_path.write_text("{}")
    kickoff_calls = []

    class FakeCrew:
        name = "Demo"
        agents = [SimpleNamespace(role="Researcher", goal="Research", backstory="")]
        tasks = [
            SimpleNamespace(
                description="Research",
                expected_output="Findings",
                output_file=None,
            )
        ]

        def kickoff(self, inputs):
            kickoff_calls.append(inputs)
            return "plain result"

    monkeypatch.setattr(
        run_crew_module, "configured_project_json_crew", lambda: crew_path
    )
    monkeypatch.setattr(
        run_crew_module,
        "_load_json_crew",
        lambda _path: (FakeCrew(), {"topic": "AI"}),
    )
    monkeypatch.setattr(
        run_crew_module,
        "_load_json_crew_for_tui",
        lambda _path: pytest.fail("DMN mode must not start the TUI loader"),
    )

    assert run_crew_module._run_json_crew() == "plain result"

    captured = capsys.readouterr()
    assert kickoff_calls == [{"topic": "AI"}]
    assert "plain result" in captured.out


def test_run_json_crew_dmn_mode_exits_on_missing_inputs(
    monkeypatch, tmp_path: Path, capsys
):
    from types import SimpleNamespace

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CREWAI_DMN", "True")
    crew_path = tmp_path / "crew.jsonc"
    crew_path.write_text("{}")
    crew = SimpleNamespace(
        agents=[
            SimpleNamespace(
                role="Researcher",
                goal="Research {topic}",
                backstory="",
            )
        ],
        tasks=[],
    )

    monkeypatch.setattr(
        run_crew_module, "configured_project_json_crew", lambda: crew_path
    )
    monkeypatch.setattr(
        run_crew_module,
        "_load_json_crew",
        lambda _path: (crew, {}),
    )

    with pytest.raises(SystemExit) as exc_info:
        run_crew_module._run_json_crew()

    captured = capsys.readouterr()
    assert exc_info.value.code == 1
    assert "Missing runtime inputs for CREWAI_DMN mode: topic" in captured.err


def test_configured_project_json_crew_defers_to_declared_flow_type(
    monkeypatch, tmp_path: Path
):
    """A flow project containing a stray crew.jsonc must still run as a flow."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "crew.jsonc").write_text("{}")
    (tmp_path / "pyproject.toml").write_text('[tool.crewai]\ntype = "flow"\n')

    assert run_crew_module.configured_project_json_crew() is None


def test_configured_project_json_crew_returns_declared_crew_definition(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    crew_path = tmp_path / "crew.jsonc"
    crew_path.write_text("{}")
    (tmp_path / "pyproject.toml").write_text(
        '[tool.crewai]\ntype = "crew"\ndefinition = "crew.jsonc"\n'
    )

    assert run_crew_module.configured_project_json_crew() == crew_path.resolve()


def test_configured_project_json_crew_ignores_declared_crew_without_definition(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "crew.jsonc").write_text("{}")
    (tmp_path / "pyproject.toml").write_text('[tool.crewai]\ntype = "crew"\n')

    assert run_crew_module.configured_project_json_crew() is None


def test_configured_project_json_crew_ignores_missing_pyproject(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "crew.jsonc").write_text("{}")

    assert run_crew_module.configured_project_json_crew() is None


def test_run_crew_inputs_without_definition_rejected_for_non_flow(monkeypatch):
    # --inputs is flow-only; in a non-flow project it now errors clearly instead
    # of the old "--inputs requires --definition".
    monkeypatch.setattr(run_crew_module, "read_toml", lambda *a, **k: {})
    monkeypatch.setattr(
        run_crew_module, "configured_project_json_crew", lambda *a, **k: None
    )
    monkeypatch.setattr(
        run_crew_module, "_warn_if_old_poetry_project", lambda *a, **k: None
    )
    monkeypatch.setattr(run_crew_module, "get_crewai_project_type", lambda *a, **k: "crew")

    with pytest.raises(click.UsageError) as exc_info:
        run_crew_module.run_crew(inputs='{"topic":"AI"}')

    assert "--inputs is only supported for declarative flows" in exc_info.value.message


def test_run_crew_inputs_without_definition_resolves_configured_flow(monkeypatch):
    # --inputs with no --definition resolves the configured [tool.crewai] flow,
    # exactly like a bare `crewai run`, and forwards the inputs.
    import crewai_cli.run_declarative_flow as rdf

    calls: dict[str, object] = {}
    monkeypatch.setattr(run_crew_module, "read_toml", lambda *a, **k: {})
    monkeypatch.setattr(
        run_crew_module, "configured_project_json_crew", lambda *a, **k: None
    )
    monkeypatch.setattr(
        run_crew_module, "_warn_if_old_poetry_project", lambda *a, **k: None
    )
    monkeypatch.setattr(run_crew_module, "get_crewai_project_type", lambda *a, **k: "flow")
    monkeypatch.setattr(
        rdf, "configured_project_declarative_flow", lambda *a, **k: Path("flow.yaml")
    )
    monkeypatch.setattr(
        rdf, "run_declarative_flow_in_project_env", lambda **kw: calls.update(kw)
    )

    run_crew_module.run_crew(inputs='{"topic":"AI"}')

    assert calls == {"definition": Path("flow.yaml"), "inputs": '{"topic":"AI"}'}


def test_run_crew_rejects_filename_with_explicit_definition():
    with pytest.raises(click.UsageError) as exc_info:
        run_crew_module.run_crew(
            trained_agents_file="trained.pkl",
            definition="flow.yaml",
        )

    assert "--filename can only be used when running crews" in exc_info.value.message


def test_run_crew_runs_explicit_declarative_definition(monkeypatch, capsys):
    calls = []

    def fake_run_declarative_flow(definition: str, inputs: str | None = None):
        calls.append((definition, inputs))

    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.run_declarative_flow",
        fake_run_declarative_flow,
    )

    run_crew_module.run_crew(definition="flow.yaml", inputs='{"topic":"AI"}')

    captured = capsys.readouterr()
    assert "experimental" not in captured.out.lower()
    assert calls == [("flow.yaml", '{"topic":"AI"}')]


def test_run_crew_runs_classic_crew_project(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(
        run_crew_module,
        "read_toml",
        lambda: {"tool": {"crewai": {"type": "crew"}}},
    )
    monkeypatch.setattr(
        run_crew_module,
        "_execute_uv_script",
        lambda script_name, **kwargs: calls.append((script_name, kwargs)),
    )

    run_crew_module.run_crew(trained_agents_file="trained.pkl")

    assert capsys.readouterr().out == ""
    assert calls == [
        (
            "run_crew",
            {"entity_type": "crew", "trained_agents_file": "trained.pkl"},
        )
    ]


def test_run_crew_runs_python_flow_project(monkeypatch, capsys):
    calls = []

    monkeypatch.setattr(
        run_crew_module,
        "read_toml",
        lambda: {"tool": {"crewai": {"type": "flow"}}},
    )
    monkeypatch.setattr(
        run_crew_module,
        "_execute_uv_script",
        lambda script_name, **kwargs: calls.append((script_name, kwargs)),
    )
    monkeypatch.setattr(
        "crewai_cli.kickoff_flow._load_conversational_flow_from_kickoff_script",
        lambda: None,
    )

    run_crew_module.run_crew()

    assert capsys.readouterr().out == ""
    assert calls == [("kickoff", {"entity_type": "flow"})]


def test_run_crew_runs_conversational_flow_tui(monkeypatch, capsys):
    class Flow:
        pass

    flow = Flow()
    calls = []

    monkeypatch.setattr(
        run_crew_module,
        "read_toml",
        lambda: {"tool": {"crewai": {"type": "flow"}}},
    )
    monkeypatch.setattr(
        "crewai_cli.kickoff_flow._load_conversational_flow_from_kickoff_script",
        lambda: flow,
    )
    monkeypatch.setattr(
        "crewai_cli.kickoff_flow._run_conversational_flow_tui",
        lambda loaded_flow: calls.append(loaded_flow),
    )
    monkeypatch.setattr(
        run_crew_module,
        "_execute_uv_script",
        lambda *_args, **_kwargs: pytest.fail(
            "conversational flows must use the TUI"
        ),
    )

    run_crew_module.run_crew()

    assert capsys.readouterr().out == ""
    assert calls == [flow]


def test_run_crew_rejects_filename_for_flow_project(monkeypatch):
    monkeypatch.setattr(
        run_crew_module,
        "read_toml",
        lambda: {"tool": {"crewai": {"type": "flow"}}},
    )

    with pytest.raises(click.UsageError) as exc_info:
        run_crew_module.run_crew(trained_agents_file="trained.pkl")

    assert "--filename can only be used when running crews" in exc_info.value.message


def test_run_crew_runs_configured_declarative_flow_project(
    monkeypatch, tmp_path: Path, capsys
):
    calls = []

    monkeypatch.chdir(tmp_path)
    definition_path = tmp_path / "flow.yaml"
    definition_path.write_text("schema: crewai.flow/v1\n", encoding="utf-8")
    monkeypatch.setattr(
        run_crew_module,
        "read_toml",
        lambda: {
            "tool": {
                "crewai": {
                    "type": "flow",
                    "definition": "flow.yaml",
                }
            }
        },
    )
    monkeypatch.setattr(
        "crewai_cli.run_declarative_flow.run_declarative_flow_in_project_env",
        lambda definition, inputs=None: calls.append((definition, inputs)),
    )
    monkeypatch.setattr(
        run_crew_module,
        "_execute_uv_script",
        lambda *_args, **_kwargs: pytest.fail("declarative flows must not run kickoff"),
    )

    run_crew_module.run_crew()

    assert capsys.readouterr().out == ""
    assert calls == [(definition_path.resolve(), None)]
