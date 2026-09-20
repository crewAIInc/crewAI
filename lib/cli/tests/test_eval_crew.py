"""`crewai eval`: the last traced run, evaluated through AMP."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import httpx
import pytest

from crewai_cli import eval_crew as eval_module
from crewai_cli.cli import eval_command


EXECUTION_ID = "6f31fe1a-20bd-4bfe-a011-25d6b9341f62"
URL = "https://evolve.crewai.test/e/ev-1"


class FakeAMP:
    """A PlusAPI double: scripted answers, calls recorded."""

    def __init__(self, create=None, statuses=None):
        self.create = create if create is not None else httpx.Response(
            202, json={"id": "ev-1", "url": URL, "status": "queued"}
        )
        self.statuses = list(statuses or [])
        self.calls: list[tuple] = []
        self.api_key = None

    def create_evaluation(self, execution_id):
        self.calls.append(("create", execution_id))
        return self.create

    def get_evaluation(self, evaluation_id):
        self.calls.append(("get", evaluation_id))
        return self.statuses.pop(0) if self.statuses else httpx.Response(200, json={"id": evaluation_id, "status": "running"})


def done(gate="passed", grades=None):
    return httpx.Response(200, json={
        "id": "ev-1", "status": "done", "url": URL,
        "verdict": {"gate": gate, "grades": grades if grades is not None else {"goal": 5, "quality": 4, "process": 5, "cost": None}},
    })


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(eval_module, "get_or_create_project_id", lambda: None)
    monkeypatch.setattr(eval_module, "saved_login", lambda: "login-token")
    monkeypatch.setattr(eval_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(eval_module, "is_dmn_mode_enabled", lambda: False)
    opened: list[str] = []
    monkeypatch.setattr(eval_module.webbrowser, "open", lambda url: opened.append(url) or True)
    return tmp_path, opened


def record_last_run(directory: Path, execution_id: str = EXECUTION_ID) -> None:
    (directory / ".crewai").mkdir(exist_ok=True)
    (directory / ".crewai" / "last_run.json").write_text(json.dumps({"execution_id": execution_id, "tier": "ephemeral"}))


def install(monkeypatch, amp: FakeAMP) -> FakeAMP:
    monkeypatch.setattr(eval_module, "PlusAPI", lambda api_key=None: (setattr(amp, "api_key", api_key), amp)[1])
    return amp


def test_the_last_run_is_evaluated_the_url_opened_and_the_verdict_printed(project, monkeypatch, capsys):
    directory, opened = project
    record_last_run(directory)
    amp = install(monkeypatch, FakeAMP(statuses=[httpx.Response(200, json={"id": "ev-1", "status": "running"}), done()]))

    eval_module.eval_crew()

    out = capsys.readouterr().out
    assert amp.api_key == "login-token"
    assert amp.calls == [("create", EXECUTION_ID), ("get", "ev-1"), ("get", "ev-1")]
    assert opened == [URL]
    assert EXECUTION_ID in out and URL in out
    assert "Goal gate: PASSED" in out and "goal 5/5" in out and "cost not measured" in out


def test_run_names_another_execution_and_an_anonymous_caller_sends_no_token(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory)
    monkeypatch.setattr(eval_module, "saved_login", lambda: None)
    amp = install(monkeypatch, FakeAMP(statuses=[done("failed")]))

    eval_module.eval_crew(run_id="other-run")

    assert amp.api_key is None
    assert amp.calls[0] == ("create", "other-run")
    assert "Goal gate: FAILED" in capsys.readouterr().out


def test_a_failed_evaluation_exits_one_with_amps_reason(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory)
    install(monkeypatch, FakeAMP(statuses=[httpx.Response(200, json={"id": "ev-1", "status": "failed", "error": "the judge was unreachable"})]))

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 1
    assert "Evaluation failed: the judge was unreachable" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (httpx.Response(401, json={"error": "account_required", "message": "Execution x was already read once without an account. Log in with `crewai login`, or create an account, to read it again."}),
         "already read once without an account"),
        (httpx.Response(401, json={"error": "bad_credentials", "message": "Bad credentials"}), "Bad credentials. Log in with `crewai login`"),
        (httpx.Response(404, json={"error": "trace_not_found", "message": "No spans recorded for execution 6f31"}), "No spans recorded for execution 6f31"),
        (httpx.Response(429, json={"error": "rate_limit_exceeded", "message": "Too many requests"}, headers={"Retry-After": "60"}), "Too many requests — retry after 60s"),
        (httpx.Response(503, json={"error": "service_unavailable", "message": "Wharf could not list the spans"}), "AMP answered 503: Wharf could not list the spans"),
        (httpx.Response(500, text="boom"), "AMP answered 500."),
    ],
)
def test_amps_refusals_are_printed_in_its_words_and_exit_one(project, monkeypatch, capsys, response, expected):
    directory, opened = project
    record_last_run(directory)
    install(monkeypatch, FakeAMP(create=response))

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 1
    assert expected in capsys.readouterr().out
    assert opened == []


def test_without_a_traced_run_and_no_terminal_it_explains_and_exits(project, monkeypatch, capsys):
    monkeypatch.setattr(eval_module, "is_dmn_mode_enabled", lambda: True)
    amp = install(monkeypatch, FakeAMP())

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 1
    out = capsys.readouterr().out
    assert "No traced run is recorded" in out and "CREWAI_TRACING_ENABLED=true" in out and "crewai run" in out
    assert amp.calls == []


def test_without_a_traced_run_it_offers_to_turn_tracing_on_and_run_the_crew(project, monkeypatch, capsys):
    directory, _ = project
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(eval_module.click, "confirm", lambda *args, **kwargs: True)
    ran: list[str] = []

    def fake_run_crew() -> None:
        ran.append("run")
        assert eval_module.os.environ.get("CREWAI_TRACING_ENABLED") == "true"
        record_last_run(directory, "fresh-run")

    import crewai_cli.run_crew as run_crew_module

    monkeypatch.setattr(run_crew_module, "run_crew", fake_run_crew)
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert ran == ["run"]
    assert "CREWAI_TRACING_ENABLED=true" in (directory / ".env").read_text()
    assert amp.calls[0] == ("create", "fresh-run")
    assert "Tracing is on for this project" in capsys.readouterr().out


def test_declining_the_offer_exits_cleanly_with_the_steps(project, monkeypatch, capsys):
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(eval_module.click, "confirm", lambda *args, **kwargs: False)
    amp = install(monkeypatch, FakeAMP())

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 0
    assert "crewai eval" in capsys.readouterr().out and amp.calls == []


def test_a_run_that_leaves_no_trace_behind_is_explained(project, monkeypatch, capsys):
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(eval_module.click, "confirm", lambda *args, **kwargs: True)
    import crewai_cli.run_crew as run_crew_module

    monkeypatch.setattr(run_crew_module, "run_crew", lambda: None)
    install(monkeypatch, FakeAMP())

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 1
    assert "no trace was recorded" in capsys.readouterr().out


def test_the_cli_command_maps_to_the_implementation(monkeypatch):
    calls = []
    monkeypatch.setattr("crewai_cli.cli.eval_crew", lambda **kwargs: calls.append(kwargs))
    runner = CliRunner()

    assert runner.invoke(eval_command, []).exit_code == 0
    assert runner.invoke(eval_command, ["--run", EXECUTION_ID]).exit_code == 0
    assert calls == [{"run_id": None}, {"run_id": EXECUTION_ID}]
    assert "Evaluate the last traced run" in runner.invoke(eval_command, ["--help"]).output


def test_last_run_id_reads_the_record_crewai_writes(tmp_path):
    assert eval_module.last_run_id(tmp_path) is None
    (tmp_path / ".crewai").mkdir()
    (tmp_path / ".crewai" / "last_run.json").write_text("not json")
    assert eval_module.last_run_id(tmp_path) is None
    record_last_run(tmp_path)
    assert eval_module.last_run_id(tmp_path) == EXECUTION_ID
