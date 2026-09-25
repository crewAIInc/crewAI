"""`crewai eval`: the last traced run, evaluated through AMP."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
import httpx
import pytest
from rich.console import Console

from crewai_cli.experimental import eval_crew as eval_module
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
    # This machine is logged in to https://amp.test (`crewai enterprise configure`).
    monkeypatch.setattr(eval_module, "Settings", lambda: SimpleNamespace(enterprise_base_url="https://amp.test"))
    monkeypatch.delenv("CREWAI_PLUS_URL", raising=False)
    opened: list[str] = []
    monkeypatch.setattr(eval_module.webbrowser, "open", lambda url: opened.append(url) or True)
    return tmp_path, opened


def record_last_run(directory: Path, execution_id: str = EXECUTION_ID, **fields) -> None:
    (directory / ".crewai").mkdir(exist_ok=True)
    record = {"execution_id": execution_id, "tier": "ephemeral", "amp_base_url": "https://amp.test", **fields}
    (directory / ".crewai" / "last_run.json").write_text(json.dumps(record))


def install(monkeypatch, amp: FakeAMP, configured_amp: str = "https://amp.test") -> FakeAMP:
    def build(api_key=None, base_url=None):
        # PlusAPI's own resolution: explicit, then CREWAI_PLUS_URL, then the saved settings.
        amp.api_key = api_key
        amp.base_url = base_url or os.environ.get("CREWAI_PLUS_URL") or configured_amp
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


def test_the_verdict_prints_whatever_areas_the_evaluation_graded(project, monkeypatch, capsys):
    # The areas are the evaluator's to name. A client printing its own list
    # would drop the ones it had not heard of and invent "not measured" for
    # ones that no longer exist — which is what happens the moment the
    # evaluation's vocabulary moves ahead of an installed CLI.
    directory, _ = project
    record_last_run(directory)
    graded = done(grades={"goal": 5, "tasks": 3, "agents": 4, "tools": None})
    install(monkeypatch, FakeAMP(statuses=[httpx.Response(200, json={"id": "ev-1", "status": "running"}), graded]))

    eval_module.eval_crew()

    out = capsys.readouterr().out
    # The whole segment, in order: asserting the parts one by one would pass
    # even if this path sorted them or printed its own list.
    assert "Goal gate: PASSED · goal 5/5 · tasks 3/5 · agents 4/5 · tools not measured" in out
    assert "quality" not in out and "process" not in out


def test_an_areas_name_is_printed_literally_never_as_markup(project, monkeypatch, capsys):
    # Every part of this line came over the wire, and a Console parses square
    # brackets. A fixed list of areas made that impossible; printing what
    # arrives does not, so an area named `[red]tasks[/red]` must show its own
    # brackets rather than restyling the verdict.
    directory, _ = project
    record_last_run(directory)
    graded = done(grades={"[red]tasks[/red]": 4, "[bold]goal": 5})
    install(monkeypatch, FakeAMP(statuses=[httpx.Response(200, json={"id": "ev-1", "status": "running"}), graded]))

    eval_module.eval_crew()

    out = capsys.readouterr().out
    assert "[red]tasks[/red] 4/5" in out
    assert "[bold]goal 5/5" in out


def test_the_follow_link_cannot_be_retargeted_by_the_url_amp_sends(project, monkeypatch, capsys):
    # This line invites a click, so a `url` carrying `[link=…]` would print a
    # trustworthy label over a hostile target. It is the worst place in this
    # command to let markup through.
    directory, _ = project
    record_last_run(directory)
    hostile = "[link=http://attacker.test/]https://app.crewai.com/e/ev-1[/link]"
    created = httpx.Response(202, json={"id": "ev-1", "url": hostile, "status": "queued"})
    install(monkeypatch, FakeAMP(create=created, statuses=[done()]))

    eval_module.eval_crew()

    out = capsys.readouterr().out
    assert "[link=http://attacker.test/]" in out  # printed, not followed


def test_a_url_that_is_not_a_string_costs_the_link_and_nothing_else(project, monkeypatch, capsys):
    # Composing the line means appending the url rather than interpolating it,
    # and `Text.append` wants a string. A malformed one must not become a
    # traceback: the evaluation is already running and its verdict is what the
    # user came for, so the link is dropped and the run carries on.
    directory, opened = project
    record_last_run(directory)
    created = httpx.Response(202, json={"id": "ev-1", "url": ["not", "a", "string"], "status": "queued"})
    install(monkeypatch, FakeAMP(create=created, statuses=[done()]))

    eval_module.eval_crew()

    out = capsys.readouterr().out
    assert "report url that is not a string" in out  # said, not swallowed
    assert "Goal gate: PASSED" in out  # and the verdict still arrives
    assert opened == []  # nothing was handed to a browser
    assert "Follow it at" not in out


def test_an_id_that_is_not_a_string_is_a_protocol_error(project, monkeypatch, capsys):
    # The id is what every later call is made with, so there is nothing to
    # carry on with — unlike the url, which is only ever shown.
    directory, _ = project
    record_last_run(directory)
    created = httpx.Response(202, json={"id": {"oops": 1}, "url": URL, "status": "queued"})
    install(monkeypatch, FakeAMP(create=created))

    with pytest.raises(SystemExit):
        eval_module.eval_crew()

    assert "without an evaluation id" in capsys.readouterr().out


def test_amps_refusal_is_printed_literally_too(project, monkeypatch, capsys):
    # The same defect on the refusal path: AMP's own sentence reaches a Console.
    directory, _ = project
    record_last_run(directory)
    refusal = httpx.Response(404, json={"error": "trace_not_found", "message": "No spans for [id]"})
    install(monkeypatch, FakeAMP(create=refusal))

    with pytest.raises(SystemExit):
        eval_module.eval_crew()

    assert "No spans for [id]" in capsys.readouterr().out


def test_an_evaluation_that_graded_nothing_prints_the_gate_alone(project, monkeypatch, capsys):
    # A well-formed verdict may carry no grades at all, and a fixed list of
    # areas used to hide that: there was always something after the separator.
    directory, _ = project
    record_last_run(directory)
    graded = done(grades={})
    install(monkeypatch, FakeAMP(statuses=[httpx.Response(200, json={"id": "ev-1", "status": "running"}), graded]))

    eval_module.eval_crew()

    out = capsys.readouterr().out
    assert "Goal gate: PASSED" in out
    assert "Goal gate: PASSED ·" not in out  # no separator with nothing after it


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

    # The project's .env is what `crewai run` traced with, so it is loaded first. This machine is
    # logged in to that AMP, so the credential goes with the request and nothing is remarked on.
    (directory / ".env").write_text("CREWAI_PLUS_URL=https://amp.test\n")
    record_last_run(directory, amp_base_url="https://amp.test/")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))
    eval_module.eval_crew()
    assert eval_module.os.environ["CREWAI_PLUS_URL"] == "https://amp.test"
    assert amp.api_key == "login-token" and amp.base_url == "https://amp.test"
    out = capsys.readouterr().out
    assert "was traced to" not in out and "Reading anonymously" not in out


def test_a_project_may_point_at_another_amp_but_never_gets_the_saved_login(project, monkeypatch, capsys):
    """A .env can send the request elsewhere — that is how a self-hosted project is wired —
    but the token goes only to an AMP this machine is logged in to."""
    directory, _ = project
    (directory / ".env").write_text("CREWAI_PLUS_URL=https://evil.example\n")
    record_last_run(directory, amp_base_url="https://evil.example")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()  # still works: AMP reads it anonymously

    assert amp.base_url == "https://evil.example"  # the request follows the project
    assert amp.api_key is None  # the credential does not
    out = capsys.readouterr().out
    assert "Reading anonymously: https://evil.example is not an AMP this machine is logged in to." in out
    assert "crewai enterprise configure" in out


def test_a_trusted_amp_over_plain_http_still_gets_no_credential(project, monkeypatch, capsys):
    """A cleartext connection is not a place to put a bearer token, trusted or not."""
    directory, _ = project
    monkeypatch.setenv("CREWAI_PLUS_URL", "http://amp.internal")  # exported, so it IS trusted
    record_last_run(directory, amp_base_url="http://amp.internal")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.base_url == "http://amp.internal" and amp.api_key is None
    assert "would carry the login over plain HTTP" in capsys.readouterr().out


def test_plain_http_to_this_machine_is_fine_for_local_development(project, monkeypatch, capsys):
    directory, _ = project
    monkeypatch.setenv("CREWAI_PLUS_URL", "http://localhost:3000")
    record_last_run(directory, amp_base_url="http://localhost:3000")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.api_key == "login-token"
    assert "Reading anonymously" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("origin", "encrypted"),
    [
        ("https://app.crewai.com", True),
        ("http://localhost:3000", True),
        ("http://127.0.0.1:8000", True),
        ("http://[::1]:8000", True),
        ("http://amp.localhost", True),
        ("http://amp.internal", False),
        ("http://169.254.169.254", False),
        ("ftp://amp.test", False),
        (None, False),
    ],
)
def test_which_connections_may_carry_the_login(origin, encrypted):
    assert eval_module._encrypted(origin) is encrypted


def test_an_amp_exported_in_this_shell_is_trusted(project, monkeypatch, capsys):
    directory, _ = project
    monkeypatch.setenv("CREWAI_PLUS_URL", "https://shell.amp.test")  # exported before the project is read
    record_last_run(directory, amp_base_url="https://shell.amp.test")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.base_url == "https://shell.amp.test" and amp.api_key == "login-token"
    assert "Reading anonymously" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("url", "origin"),
    [
        ("https://amp.test", "https://amp.test"),
        ("https://AMP.Test/crewai_plus/", "https://amp.test"),
        ("http://localhost:8000/x", "http://localhost:8000"),
        ("app.crewai.com", None),  # no scheme: not an origin, never trusted
        ("", None),
        (None, None),
    ],
)
def test_an_origin_is_scheme_and_host_only(url, origin):
    assert eval_module._origin(url) == origin


@pytest.mark.parametrize(
    "verdict",
    [
        None,
        "passed",
        [],
        {"gate": "passed"},
        {"gate": None, "grades": {}},
        {"gate": "passed", "grades": "5/5"},
        {"gate": "passed", "grades": {"goal": "five"}},
        {"gate": "passed", "grades": {"goal": True}},  # a bool is an int to Python, never a grade
        {"gate": "passed", "grades": {"goal": 6}},  # out of the 1..5 range
        {"gate": "passed", "grades": {"goal": 0}},
        {"gate": "passed", "grades": {"goal": 4.5}},
    ],
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
    # the offer is about what happens, not about the variable that makes it happen
    assert "Turn tracing on and run the crew now?" in text and kwargs == {"default": True}  # Enter is yes (João's call)
    assert "sent to CrewAI AMP" in text  # what saying yes means, in the words that matter
    assert "CREWAI_TRACING_ENABLED" not in text  # and not the variable that carries it
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


def test_only_a_missing_login_reads_as_anonymous(monkeypatch, capsys):
    """An unreadable credential store is not "anonymous": it is said out loud."""
    from crewai_cli.authentication.token import AuthError

    monkeypatch.setattr(eval_module, "get_auth_token", lambda: (_ for _ in ()).throw(AuthError("No token found")))
    assert eval_module.saved_login() is None  # not logged in: AMP treats the caller as anonymous

    monkeypatch.setattr(eval_module, "get_auth_token", lambda: "login-token")
    assert eval_module.saved_login() == "login-token"

    for broken in (OSError(13, "Permission denied"), ValueError("Fernet key must be 32 url-safe base64-encoded bytes.")):
        monkeypatch.setattr(eval_module, "get_auth_token", lambda error=broken: (_ for _ in ()).throw(error))
        with pytest.raises(SystemExit) as exit_:
            eval_module.saved_login()
        assert exit_.value.code == 1
        out = capsys.readouterr().out
        assert "Could not read the saved login" in out and type(broken).__name__ in out and "crewai login" in out


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
