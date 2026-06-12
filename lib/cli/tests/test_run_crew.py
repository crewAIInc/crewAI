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

    def fake_run_json_crew(daemon=False, trained_agents_file=None):
        called["daemon"] = daemon
        called["trained_agents_file"] = trained_agents_file

    monkeypatch.setattr(run_crew_module, "_run_json_crew", fake_run_json_crew)

    run_crew_module.run_crew(trained_agents_file="some.pkl", daemon=True)

    assert called == {"daemon": True, "trained_agents_file": "some.pkl"}


def test_run_json_crew_exports_trained_agents_env(monkeypatch, tmp_path: Path):
    """JSON crews run in-process, so the pickle path must land in the env var."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CREWAI_TRAINED_AGENTS_FILE_ENV, raising=False)

    try:
        # No crew.json(c) in tmp_path: the loader fails *after* the env var
        # export, which is the part under test.
        with pytest.raises(FileNotFoundError):
            run_crew_module._run_json_crew(
                daemon=True, trained_agents_file="some.pkl"
            )
        assert os.environ[CREWAI_TRAINED_AGENTS_FILE_ENV] == "some.pkl"
    finally:
        os.environ.pop(CREWAI_TRAINED_AGENTS_FILE_ENV, None)


def test_run_json_crew_leaves_env_untouched_without_flag(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(CREWAI_TRAINED_AGENTS_FILE_ENV, raising=False)

    with pytest.raises(FileNotFoundError):
        run_crew_module._run_json_crew(daemon=True)

    assert CREWAI_TRAINED_AGENTS_FILE_ENV not in os.environ
