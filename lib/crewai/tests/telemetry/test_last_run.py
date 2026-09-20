"""`.crewai/last_run.json`: the run `crewai eval` evaluates by default."""

from __future__ import annotations

import json

import pytest

from crewai.telemetry.tracing import last_run


@pytest.fixture
def project(monkeypatch, tmp_path):
    monkeypatch.setattr(last_run, "project_dir", lambda: tmp_path)
    monkeypatch.setattr(last_run, "recording_enabled", lambda: True)
    return tmp_path


def test_a_run_is_recorded_atomically_and_read_back(project):
    path = last_run.record_last_run(
        execution_id="6f31fe1a-20bd-4bfe-a011-25d6b9341f62",
        tier="ephemeral",
        started_at_ns=1_758_240_000_000_000_000,
        finished_at_ns=1_758_240_009_500_000_000,
        amp_base_url="https://app.crewai.com",
    )
    assert path == project / ".crewai" / "last_run.json"
    assert not path.with_name("last_run.json.tmp").exists()
    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["execution_id"] == "6f31fe1a-20bd-4bfe-a011-25d6b9341f62"
    assert written["tier"] == "ephemeral"
    assert written["started_at"] == "2025-09-19T00:00:00.000+00:00"
    assert written["finished_at"] == "2025-09-19T00:00:09.500+00:00"
    assert written["amp_base_url"] == "https://app.crewai.com"
    assert written["recorded_at"]
    assert last_run.read_last_run(project) == written


def test_the_newest_run_replaces_the_previous_one(project):
    last_run.record_last_run(execution_id="first", tier=None, started_at_ns=None, finished_at_ns=None, amp_base_url=None)
    last_run.record_last_run(execution_id="second", tier="authenticated", started_at_ns=None, finished_at_ns=None, amp_base_url=None)
    written = last_run.read_last_run(project)
    assert written is not None and written["execution_id"] == "second"
    assert written["started_at"] is None and written["finished_at"] is None


def test_nothing_is_recorded_under_the_test_suite(monkeypatch, tmp_path):
    monkeypatch.setattr(last_run, "project_dir", lambda: tmp_path)
    monkeypatch.setenv("CREWAI_TESTING", "true")
    assert last_run.record_last_run(execution_id="x", tier=None, started_at_ns=None, finished_at_ns=None, amp_base_url=None) is None
    assert not (tmp_path / ".crewai").exists()


def test_a_missing_or_broken_record_reads_as_none(project):
    assert last_run.read_last_run(project) is None
    (project / ".crewai").mkdir()
    (project / ".crewai" / "last_run.json").write_text("not json", encoding="utf-8")
    assert last_run.read_last_run(project) is None
    (project / ".crewai" / "last_run.json").write_text(json.dumps({"tier": "ephemeral"}), encoding="utf-8")
    assert last_run.read_last_run(project) is None  # no execution id: no run


def test_a_write_failure_never_raises(project, monkeypatch):
    (project / ".crewai").write_text("a file where the directory should be", encoding="utf-8")
    assert last_run.record_last_run(execution_id="x", tier=None, started_at_ns=None, finished_at_ns=None, amp_base_url=None) is None
