"""`crewai eval`: the last traced run, evaluated through AMP."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner
import httpx
import pytest
from rich.console import Console

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
    monkeypatch.setattr(eval_module, "console", Console(width=240))  # one sentence per line in the captured output
    monkeypatch.setattr(eval_module, "get_or_create_project_id", lambda: None)
    monkeypatch.setattr(eval_module, "saved_login", lambda: "login-token")
    monkeypatch.setattr(eval_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(eval_module, "is_dmn_mode_enabled", lambda: False)
    opened: list[str] = []
    monkeypatch.setattr(eval_module.webbrowser, "open", lambda url: opened.append(url) or True)
    return tmp_path, opened


def record_last_run(directory: Path, execution_id: str = EXECUTION_ID, **fields) -> None:
    (directory / ".crewai").mkdir(exist_ok=True)
    record = {"execution_id": execution_id, "tier": "ephemeral", "amp_base_url": "https://amp.test", **fields}
    (directory / ".crewai" / "last_run.json").write_text(json.dumps(record))


def install(monkeypatch, amp: FakeAMP, configured_amp: str = "https://amp.test") -> FakeAMP:
    def build(api_key=None, base_url=None):
        amp.api_key, amp.base_url = api_key, base_url or configured_amp
        return amp

    monkeypatch.setattr(eval_module, "PlusAPI", build)
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
        (httpx.Response(404, text="<html>Page not found</html>"), f"AMP answered 404 for run {EXECUTION_ID}."),
        (httpx.Response(202, json={"url": URL}), "AMP answered without an evaluation id (202)."),
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


def test_the_credential_goes_only_to_the_configured_amp_never_to_an_address_off_the_record(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory, amp_base_url="https://evil.example/steal")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]), configured_amp="https://app.crewai.com")

    eval_module.eval_crew()

    assert amp.api_key == "login-token" and amp.base_url == "https://app.crewai.com"  # PlusAPI got no base_url
    out = capsys.readouterr().out
    assert "The run was traced to https://evil.example/steal; evaluating at the configured AMP https://app.crewai.com." in out

    # The project's .env is what `crewai run` traced with, so it is loaded first: same AMP, no note.
    (directory / ".env").write_text("CREWAI_PLUS_URL=https://amp.test\n")
    record_last_run(directory, amp_base_url="https://amp.test/")
    install(monkeypatch, FakeAMP(statuses=[done()]))
    monkeypatch.delenv("CREWAI_PLUS_URL", raising=False)
    eval_module.eval_crew()
    assert eval_module.os.environ["CREWAI_PLUS_URL"] == "https://amp.test"
    assert "was traced to" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "verdict",
    [None, "passed", [], {"gate": "passed"}, {"gate": None, "grades": {}}, {"gate": "passed", "grades": "5/5"}, {"gate": "passed", "grades": {"goal": "five"}}],
)
def test_a_done_answer_without_a_well_formed_verdict_is_a_protocol_error(project, monkeypatch, capsys, verdict):
    directory, _ = project
    record_last_run(directory)
    body = {"id": "ev-1", "status": "done", "url": URL}
    if verdict is not None:
        body["verdict"] = verdict
    install(monkeypatch, FakeAMP(statuses=[httpx.Response(200, json=body)]))

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 1
    assert f"AMP answered done without a verdict (protocol error); follow it at {URL}." in capsys.readouterr().out


def test_amp_unreachable_at_the_start_is_a_sentence_not_a_traceback(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory)
    amp = install(monkeypatch, FakeAMP())
    monkeypatch.setattr(amp, "create_evaluation", lambda execution_id: (_ for _ in ()).throw(httpx.ConnectError("connection refused")))

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 1
    assert "Could not reach AMP to start the evaluation: connection refused" in capsys.readouterr().out


def test_while_waiting_a_blip_is_retried_and_a_streak_is_reported(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory)
    amp = install(monkeypatch, FakeAMP(statuses=[httpx.Response(502), httpx.Response(503, json={"error": "service_unavailable", "message": "crew-optimize is down"}), done()]))

    eval_module.eval_crew()  # two bad polls, then the verdict

    assert "Goal gate: PASSED" in capsys.readouterr().out
    assert amp.calls.count(("get", "ev-1")) == 3

    record_last_run(directory)
    amp = install(monkeypatch, FakeAMP(statuses=[httpx.Response(503, json={"error": "service_unavailable", "message": "crew-optimize is down"})] * eval_module.POLL_RETRIES))
    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()
    assert exit_.value.code == 1
    assert "AMP answered 503: crew-optimize is down" in capsys.readouterr().out
    assert amp.calls.count(("get", "ev-1")) == eval_module.POLL_RETRIES

    amp = install(monkeypatch, FakeAMP())
    monkeypatch.setattr(amp, "get_evaluation", lambda evaluation_id: (_ for _ in ()).throw(httpx.ReadTimeout("timed out")))
    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()
    assert exit_.value.code == 1
    assert f"Could not reach AMP while waiting (timed out); the evaluation keeps running at {URL}." in capsys.readouterr().out


def test_a_refusal_mid_poll_names_the_evaluation_and_an_unknown_status_stops_the_wait(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory)
    install(monkeypatch, FakeAMP(statuses=[httpx.Response(404, text="gone")]))
    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()
    assert exit_.value.code == 1 and "AMP answered 404 for evaluation ev-1." in capsys.readouterr().out

    for odd in (httpx.Response(200, json={"id": "ev-1", "status": "cancelled"}), httpx.Response(200, json=[]), httpx.Response(200, text="<html>")):
        install(monkeypatch, FakeAMP(statuses=[httpx.Response(200, json={"id": "ev-1", "status": "queued"}), odd]))
        with pytest.raises(SystemExit) as exit_:
            eval_module.eval_crew()
        assert exit_.value.code == 1
        assert f"AMP answered without a known evaluation status" in capsys.readouterr().out


def test_ctrl_c_leaves_the_evaluation_running_and_exits_130(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory)
    amp = install(monkeypatch, FakeAMP())
    monkeypatch.setattr(amp, "get_evaluation", lambda evaluation_id: (_ for _ in ()).throw(KeyboardInterrupt()))

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 130
    assert f"Still running at {URL}." in capsys.readouterr().out


def test_dmn_mode_prints_the_url_but_opens_no_browser(project, monkeypatch, capsys):
    directory, opened = project
    record_last_run(directory)
    monkeypatch.setattr(eval_module, "is_dmn_mode_enabled", lambda: True)
    install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert opened == [] and URL in capsys.readouterr().out


def test_run_skips_the_offer_when_nothing_is_recorded(project, monkeypatch, capsys):
    monkeypatch.setattr(eval_module.click, "confirm", lambda *args, **kwargs: pytest.fail("no offer with --run"))
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew(run_id="named-run")

    assert amp.calls[0] == ("create", "named-run")


def test_outside_a_crewai_project_nothing_is_written_and_it_says_so(project, monkeypatch, capsys):
    directory, _ = project
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(eval_module.click, "confirm", lambda *args, **kwargs: pytest.fail("no offer outside a project"))
    amp = install(monkeypatch, FakeAMP())

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 1
    assert "No crewAI project here (no pyproject.toml)" in capsys.readouterr().out
    assert not (directory / ".env").exists() and amp.calls == []


def test_without_a_traced_run_and_no_terminal_it_explains_and_exits(project, monkeypatch, capsys):
    (project[0] / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
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
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    prompts: list[tuple] = []
    monkeypatch.setattr(eval_module.click, "confirm", lambda text, **kwargs: prompts.append((text, kwargs)) or True)
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
    text, kwargs = prompts[0]
    assert "CREWAI_TRACING_ENABLED=true stays in .env" in text and kwargs == {"default": True}  # Enter is yes (João's call); the prompt names both effects
    assert "CREWAI_TRACING_ENABLED=true" in (directory / ".env").read_text()
    assert amp.calls[0] == ("create", "fresh-run")
    assert "Tracing is on for this project" in capsys.readouterr().out


def test_declining_the_offer_exits_cleanly_with_the_steps(project, monkeypatch, capsys):
    (project[0] / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(eval_module.click, "confirm", lambda *args, **kwargs: False)
    amp = install(monkeypatch, FakeAMP())

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 0
    assert "crewai eval" in capsys.readouterr().out and amp.calls == []


def test_a_run_that_leaves_no_trace_behind_is_explained(project, monkeypatch, capsys):
    (project[0] / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(eval_module.click, "confirm", lambda *args, **kwargs: True)
    import crewai_cli.run_crew as run_crew_module

    monkeypatch.setattr(run_crew_module, "run_crew", lambda: None)
    install(monkeypatch, FakeAMP())

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 1
    out = capsys.readouterr().out
    assert "no trace was recorded" in out and "older than the version that records the last run" in out


def test_the_cli_command_maps_to_the_implementation(monkeypatch):
    calls = []
    monkeypatch.setattr("crewai_cli.cli.eval_crew", lambda **kwargs: calls.append(kwargs))
    runner = CliRunner()

    assert runner.invoke(eval_command, []).exit_code == 0
    assert runner.invoke(eval_command, ["--run", EXECUTION_ID]).exit_code == 0
    assert calls == [{"run_id": None}, {"run_id": EXECUTION_ID}]
    assert "Evaluate the last traced run" in runner.invoke(eval_command, ["--help"]).output


def test_read_last_run_reads_the_record_crewai_writes(tmp_path):
    assert eval_module.read_last_run(tmp_path) is None
    (tmp_path / ".crewai").mkdir()
    (tmp_path / ".crewai" / "last_run.json").write_text("not json")
    assert eval_module.read_last_run(tmp_path) is None
    (tmp_path / ".crewai" / "last_run.json").write_text(json.dumps({"tier": "ephemeral"}))
    assert eval_module.read_last_run(tmp_path) is None
    record_last_run(tmp_path)
    record = eval_module.read_last_run(tmp_path)
    assert record is not None and record["execution_id"] == EXECUTION_ID and record["amp_base_url"] == "https://amp.test"
