"""Tests for crewai_cli.run_crew JSON crew handling."""

import os
from pathlib import Path

import pytest
from crewai_core.constants import CREWAI_TRAINED_AGENTS_FILE_ENV

import crewai_cli.run_crew as run_crew_module


def test_run_crew_forwards_trained_agents_file_to_json_crews(monkeypatch):
    """crewai run -f must reach JSON crews, not only classic subprocess crews."""
    monkeypatch.setattr(run_crew_module, "_has_json_crew", lambda: True)
    called: dict = {}

    def fake_run_json_crew(trained_agents_file=None):
        called["trained_agents_file"] = trained_agents_file

    monkeypatch.setattr(run_crew_module, "_run_json_crew", fake_run_json_crew)

    run_crew_module.run_crew(trained_agents_file="some.pkl")

    assert called == {"trained_agents_file": "some.pkl"}


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
