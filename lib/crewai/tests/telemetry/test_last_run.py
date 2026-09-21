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
    assert [child.name for child in path.parent.iterdir()] == ["last_run.json"]  # no temporary file left behind
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


def test_nothing_is_recorded_inside_a_deployment(monkeypatch, tmp_path):
    """The platform's integration token marks a deployment: no file the platform never reads."""
    monkeypatch.setattr(last_run, "project_dir", lambda: tmp_path)
    monkeypatch.delenv("CREWAI_TESTING", raising=False)
    monkeypatch.setenv("CREWAI_PLATFORM_INTEGRATION_TOKEN", "platform-token")
    assert last_run.record_last_run(execution_id="x", tier="authenticated", started_at_ns=None, finished_at_ns=None, amp_base_url=None) is None
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


def test_each_writer_uses_a_temporary_file_of_its_own(project, monkeypatch):
    """Two crews finishing together in one project: neither may replace the other's temporary file."""
    replaced: list[str] = []
    real_replace = last_run.os.replace
    monkeypatch.setattr(last_run.os, "replace", lambda src, dst: replaced.append(str(src)) or real_replace(src, dst))
    for execution_id in ("first", "second"):
        last_run.record_last_run(execution_id=execution_id, tier=None, started_at_ns=None, finished_at_ns=None, amp_base_url=None)
    assert len(replaced) == 2 and replaced[0] != replaced[1]
    names = [source.rsplit("/", 1)[-1] for source in replaced]
    assert all(name.startswith(".last_run.json.") and name.endswith(".tmp") for name in names)
    assert [child.name for child in (project / ".crewai").iterdir()] == ["last_run.json"]


def test_a_failed_write_leaves_no_temporary_file(project, monkeypatch):
    monkeypatch.setattr(last_run.os, "replace", lambda src, dst: (_ for _ in ()).throw(OSError("disk full")))
    assert last_run.record_last_run(execution_id="x", tier=None, started_at_ns=None, finished_at_ns=None, amp_base_url=None) is None
    assert list((project / ".crewai").iterdir()) == []


def test_the_run_that_finished_last_stays_recorded_whichever_writer_comes_last(project):
    """Two crews finish together; the OLDER run's writer gets to the file after the newer one did."""
    second = 1_000_000_000
    base = 1_758_240_000 * second
    last_run.record_last_run(execution_id="newer", tier="ephemeral", started_at_ns=base, finished_at_ns=base + 30 * second, amp_base_url=None)
    kept = last_run.record_last_run(execution_id="older", tier="ephemeral", started_at_ns=base, finished_at_ns=base + 10 * second, amp_base_url=None)
    written = last_run.read_last_run(project)
    assert kept == project / ".crewai" / "last_run.json"
    assert written is not None and written["execution_id"] == "newer"
    assert [child.name for child in (project / ".crewai").iterdir()] == ["last_run.json"]  # the loser's temporary file is gone

    # The same run recorded again (a refreshed grant) and a run that finished later both replace it.
    last_run.record_last_run(execution_id="newer", tier="authenticated", started_at_ns=base, finished_at_ns=base + 30 * second, amp_base_url=None)
    assert last_run.read_last_run(project)["tier"] == "authenticated"
    last_run.record_last_run(execution_id="newest", tier="ephemeral", started_at_ns=base, finished_at_ns=base + 40 * second, amp_base_url=None)
    assert last_run.read_last_run(project)["execution_id"] == "newest"

    # A tie to the millisecond keeps what is there: the file orders runs no finer than it stores them.
    last_run.record_last_run(execution_id="same-instant", tier="ephemeral", started_at_ns=base, finished_at_ns=base + 40 * second + 400_000, amp_base_url=None)
    assert last_run.read_last_run(project)["execution_id"] == "newest"

    # A record without a comparable time never blocks the run just finished.
    (project / ".crewai" / "last_run.json").write_text(json.dumps({"execution_id": "legacy"}), encoding="utf-8")
    last_run.record_last_run(execution_id="fresh", tier=None, started_at_ns=None, finished_at_ns=None, amp_base_url=None)
    assert last_run.read_last_run(project)["execution_id"] == "fresh"
