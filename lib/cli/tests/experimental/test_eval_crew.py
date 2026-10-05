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

from crewai_cli import run_crew as run_crew_module
from crewai_cli.experimental import eval_crew as eval_module
from crewai_cli.cli import eval_command


@pytest.fixture(autouse=True)
def _awaiting_files_stay_in_tmp(tmp_path, monkeypatch):
    """The waiting-evaluation token file belongs to the user's crewAI data
    directory; a test keeps it in its own."""
    monkeypatch.setattr(
        "crewai_cli.crew_run_tui._awaiting_dir", lambda: tmp_path / "eval-awaiting"
    )



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
        self.sent_config = None
        self.api_key = None
        self.headers: dict[str, str] = {"X-Crewai-Organization-Id": "org-42"}

    def create_evaluation(self, execution_id, *, eval_config=None, project_id=None):
        self.calls.append(("create", execution_id))
        self.sent_config = eval_config
        self.sent_project_id = project_id
        if isinstance(self.create, list):
            return self.create.pop(0) if len(self.create) > 1 else self.create[0]
        return self.create

    def get_evaluation(self, evaluation_id):
        self.calls.append(("get", evaluation_id))
        return self.statuses.pop(0) if self.statuses else httpx.Response(200, json={"id": evaluation_id, "status": "running"})


def done(gate="passed", grades=None, eval_config=None):
    payload = {
        "id": "ev-1", "status": "done", "url": URL,
        "verdict": {"gate": gate, "grades": grades if grades is not None else {"goal": 5, "quality": 4, "process": 5, "cost": None}},
    }
    if eval_config is not None:
        payload["eval_config"] = eval_config
    return httpx.Response(200, json=payload)


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


def _now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


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
    # A web address that carries markup inside it: printed as text, never
    # interpreted (one that IS markup is not a web address, and is dropped).
    hostile = "https://app.crewai.com/e/[link=http://attacker.test/]ev-1[/link]"
    created = httpx.Response(202, json={"id": "ev-1", "url": hostile, "status": "queued"})
    install(monkeypatch, FakeAMP(create=created, statuses=[done()]))

    eval_module.eval_crew()

    out = capsys.readouterr().out.replace("\n", "")
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
    assert "not a web address" in out  # said, not swallowed
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

    with pytest.raises(SystemExit):  # a failed gate is exit 1
        eval_module.eval_crew(run_id="other-run")

    assert amp.api_key is None
    assert amp.calls[0] == ("create", "other-run")
    assert "Goal gate: FAILED" in capsys.readouterr().out


@pytest.mark.parametrize(
    "gate, code", [("passed", None), ("failed", 1), ("inconclusive", 1), ("unknown", 1)]
)
def test_the_exit_code_is_the_gates(project, monkeypatch, capsys, gate, code):
    """A CI job reads the exit code. Only a gate that PASSED is 0: a failed one,
    and one with no verdict, must stop the pipeline rather than wave it on."""
    directory, _ = project
    record_last_run(directory)
    install(monkeypatch, FakeAMP(statuses=[done(gate)]))

    if code is None:
        eval_module.eval_crew()
    else:
        with pytest.raises(SystemExit) as exited:
            eval_module.eval_crew()
        assert exited.value.code == code
    assert f"Goal gate: {gate.upper()}" in capsys.readouterr().out


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


def test_the_app_never_trusts_an_amp_the_project_env_introduced(project, monkeypatch):
    """`crewai eval` reads CREWAI_PLUS_URL before the project's `.env` is
    loaded, so a shell export is trusted there and a project cannot slip one
    in. Inside the run app there is no such "before" — the crew has already run
    and its `.env` was loaded for it — so an exported variable buys nothing and
    the saved login stays home."""
    directory, _ = project
    # what a project's .env would leave behind, indistinguishable by then
    monkeypatch.setenv("CREWAI_PLUS_URL", "https://evil.example")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))
    notes: list[str] = []

    eval_module.evaluate_run(
        EXECUTION_ID, on_started=lambda started: None, note=notes.append
    )

    assert amp.base_url == "https://evil.example"  # the request still follows it
    assert amp.api_key is None  # the credential does not
    assert any("Reading anonymously" in note for note in notes)
    # and it is said through the caller's own note, never printed over its screen
    assert notes and all(isinstance(note, str) for note in notes)


def test_the_command_still_trusts_an_amp_exported_in_the_shell(project, monkeypatch, capsys):
    """The command's own reading is unchanged: a shell export is this machine
    speaking, and the login goes with it."""
    directory, _ = project
    monkeypatch.setenv("CREWAI_PLUS_URL", "https://amp.exported")
    record_last_run(directory, amp_base_url="https://amp.exported")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.base_url == "https://amp.exported"
    assert amp.api_key == "login-token"
    assert "Reading anonymously" not in capsys.readouterr().out


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
    monkeypatch.setattr(amp, "create_evaluation", lambda execution_id, **_: (_ for _ in ()).throw(httpx.ConnectError("connection refused")))

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
        eval_module.record_evaluation_outcome("fresh-run")  # what the app does

    import crewai_cli.run_crew as run_crew_module

    monkeypatch.setattr(run_crew_module, "run_crew", fake_run_crew)
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
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
    # the app that ran the crew evaluates it on its own screen; the command that
    # opened it does not ask AMP a second time
    assert amp.calls == []
    assert "Tracing is on for this project" in capsys.readouterr().out


@pytest.mark.parametrize("logged_in", [True, False])
def test_the_command_counts_the_evaluation_and_nothing_that_names_it(
    project, monkeypatch, logged_in
):
    """`cli_usage:eval` is every evaluation; the TUI's own `cli_usage:evaluate`
    is intent, and the difference is intent that never became one. Usage stats
    are anonymous, so the span says whether the caller was logged in — never
    which run, which organization, or anything about the run's content."""
    directory, _ = project
    record_last_run(directory, "counted-run")
    spans: list[tuple[str, dict[str, str]]] = []

    class FakeTelemetry:
        def set_tracer(self) -> None:
            pass

        def feature_usage_span(self, feature, attributes=None) -> None:
            spans.append((feature, attributes or {}))

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", FakeTelemetry)
    monkeypatch.setattr(eval_module, "saved_login", lambda: "tok" if logged_in else None)
    install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert spans == [
        ("cli_usage:eval", {"authenticated": "true" if logged_in else "false"})
    ]


def test_an_oversized_config_is_said_once_and_through_the_callers_note(
    project, monkeypatch
):
    """The run app routes notes onto its own screen, and the spans of a fresh
    run can take several 404s to land — the warning belongs there, once."""
    directory, _ = project
    (directory / "eval.jsonc").write_text("x" * (eval_module.MAX_EVAL_CONFIG_BYTES + 1))
    sent: list[str | None] = []
    answers = iter([httpx.Response(404, json={}), httpx.Response(404, json={}), QUEUED])

    class Client:
        def create_evaluation(self, execution_id, *, eval_config=None, project_id=None):
            sent.append(eval_config)
            return next(answers)

    monkeypatch.setattr(eval_module.time, "sleep", lambda _s: None)
    notes: list[str] = []

    eval_module._start_evaluation(
        Client(), "run-1", wait_for_spans=True, note=notes.append
    )

    assert sent == [None, None, None]
    assert [n for n in notes if "larger than" in n] == [notes[0]]


def test_the_fallback_grades_only_a_run_recorded_after_it_began(
    project, monkeypatch, capsys
):
    """No app evaluated the run, so the command reads the project's record — but
    that record is the project's LAST run, and one written before this run
    began belongs to some other run."""
    directory, _ = project
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        run_crew_module,
        "run_crew",
        lambda: record_last_run(
            directory, "somebody-elses-run", recorded_at="2020-01-01T00:00:00+00:00"
        ),
    )
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    with pytest.raises(SystemExit) as exited:
        eval_module.eval_crew()

    assert exited.value.code == 1
    assert amp.calls == []
    assert "no trace was recorded" in capsys.readouterr().out


@pytest.mark.parametrize(
    "url, opened",
    [
        ("https://optimize.crewai.test/e/ev-1", True),
        ("http://localhost:3000/e/ev-1", True),
        ("http://evil.test/e/ev-1", False),
        ("file:///etc/passwd", False),
        ("javascript:alert(1)", False),
        ("https://user:pw@optimize.crewai.test/e/ev-1", False),
        ("https://optimize.crewai.test/e/ev-1\x1b[31m", False),
        ("/e/ev-1", False),
    ],
)
def test_only_a_web_address_is_printed_and_opened(url, opened):
    """The report link comes from whichever AMP answered — which a project's
    `.env` may choose — and is opened without a click. Its host is not pinned
    (the report lives on the evaluation service), but it must be a web address."""
    assert (eval_module._report_url(url) == url) is opened


def test_a_report_url_that_is_not_a_web_address_is_dropped_with_a_note(monkeypatch):
    class Client:
        def create_evaluation(self, execution_id, *, eval_config=None, project_id=None):
            return httpx.Response(
                202, json={"id": "ev-1", "url": "file:///etc/passwd", "status": "queued"}
            )

    notes: list[str] = []
    started = eval_module._start_evaluation(Client(), "run-1", note=notes.append)

    assert started["url"] is None
    assert "link is unavailable" in notes[0]


@pytest.mark.parametrize("login", [None, "tok"])
def test_a_request_without_the_login_names_no_organization(project, monkeypatch, login):
    """AMP reads the organization header only beside a credential, so without
    one it says nothing — except, to an AMP this machine is not logged in to,
    which organization is asking."""
    monkeypatch.setenv("CREWAI_PLUS_URL", "https://not-logged-in.test")
    monkeypatch.setattr(eval_module, "saved_login", lambda: login)
    monkeypatch.setattr(
        "crewai_core.plus_api.Settings",
        lambda: SimpleNamespace(org_uuid="org-42", enterprise_base_url=None),
    )

    client = eval_module._amp_client({"https://app.crewai.com"}, note=lambda _t: None)

    assert client.api_key is None
    assert "X-Crewai-Organization-Id" not in client.headers


def test_telemetry_never_breaks_the_command(project, monkeypatch):
    directory, _ = project
    record_last_run(directory, "counted-run")

    def explode() -> None:
        raise RuntimeError("no tracer here")

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", lambda: explode())
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.calls[0] == ("create", "counted-run")


def test_a_project_that_says_what_good_means_is_graded_on_it(project, monkeypatch):
    """The file travels as it was written — comments and all — because what a
    criterion means is the grader's to read, not this command's."""
    directory, _ = project
    record_last_run(directory)
    written = '// ours\n{"dataset": [{"id": "c1", "criteria": ["cites a source"]}]}\n'
    (directory / "eval.jsonc").write_text(written)
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.sent_config == written


def test_a_project_with_nothing_to_say_sends_nothing(project, monkeypatch):
    directory, _ = project
    record_last_run(directory)
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.sent_config is None


def test_a_config_too_large_to_be_criteria_is_left_behind(project, monkeypatch, capsys):
    """A file that AMP would refuse is not uploaded, and the run is still graded."""
    directory, _ = project
    record_last_run(directory)
    (directory / "eval.jsonc").write_text("// " + "x" * (64 * 1024))
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.sent_config is None
    assert amp.calls[0][0] == "create"  # graded anyway
    assert "was not sent" in capsys.readouterr().out


def test_the_first_evaluation_leaves_the_criteria_behind_as_a_file(
    project, monkeypatch, capsys
):
    """A project cannot say what good means until it has somewhere to say it."""
    directory, _ = project
    record_last_run(directory)
    install(monkeypatch, FakeAMP(statuses=[done(eval_config='// yours\n{"dataset": []}\n')]))

    eval_module.eval_crew()

    assert (directory / "eval.jsonc").read_text() == '// yours\n{"dataset": []}\n'
    assert "Wrote eval.jsonc" in capsys.readouterr().out


def test_a_file_a_project_already_has_is_never_overwritten(project, monkeypatch, capsys):
    """After the first one the file is theirs — the one thing worse than no
    criteria is criteria that vanish every time somebody evaluates."""
    directory, _ = project
    record_last_run(directory)
    (directory / "eval.jsonc").write_text("// mine, edited\n{\"dataset\": []}\n")
    install(monkeypatch, FakeAMP(statuses=[done(eval_config='// theirs\n{}')]))

    eval_module.eval_crew()

    assert (directory / "eval.jsonc").read_text() == "// mine, edited\n{\"dataset\": []}\n"
    assert "Wrote eval.jsonc" not in capsys.readouterr().out


def _write_marker(directory: Path, execution_id: str | None, at: str) -> None:
    (directory / ".crewai").mkdir(exist_ok=True)
    (directory / ".crewai" / "last_eval.json").write_text(
        json.dumps({"execution_id": execution_id, "at": at})
    )


def _seconds_ago(seconds: float) -> str:
    from datetime import datetime, timedelta, timezone

    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


NOT_FOUND = httpx.Response(
    404,
    json={
        "error": "trace_not_found",
        "message": "No spans recorded for execution 6f31fe1a",
    },
)
QUEUED = httpx.Response(202, json={"id": "ev-1", "url": URL, "status": "queued"})


def test_the_app_it_opens_is_told_an_evaluation_is_waiting(project, monkeypatch):
    """With nothing traced, the command runs the crew — and the app it opens is
    told to evaluate what it ran, so nobody quits a screen to reach it."""
    directory, _ = project
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)

    def fake_run_crew() -> None:
        from crewai_cli.crew_run_tui import _AUTO_EVAL

        assert _AUTO_EVAL.get() is not None, "the app must be told in this process"
        # and in a child process, where a ContextVar cannot reach: a token, and
        # the file of that name that makes it count
        from crewai_cli.crew_run_tui import _awaiting_dir

        assert (_awaiting_dir() / os.environ["CREWAI_EVAL_AWAITING_RUN"]).is_file()
        record_last_run(directory, "run-it-just-did")
        eval_module.record_evaluation_outcome("run-it-just-did")

    monkeypatch.setattr(run_crew_module, "run_crew", fake_run_crew)
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.calls == []  # the app did it, on its own screen
    assert "CREWAI_EVAL_AWAITING_RUN" not in os.environ
    from crewai_cli.crew_run_tui import _awaiting_dir

    assert not any(_awaiting_dir().iterdir())  # and the token went with it


def test_a_run_no_app_evaluated_is_graded_by_the_command(project, monkeypatch):
    """Not every run goes through the app: a conversational session never ends
    the way a crew does, and a flow with human feedback takes the terminal
    instead. Nothing marked the run as evaluated, so the command grades it
    rather than exiting as if somebody had."""
    directory, _ = project
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        run_crew_module,
        "run_crew",
        lambda: record_last_run(directory, "run-nobody-graded", recorded_at=_now()),
    )
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.calls[0] == ("create", "run-nobody-graded")


def test_a_marker_from_an_earlier_run_never_counts_for_this_one(project, monkeypatch):
    """A project keeps one marker, and the one from an earlier `crewai eval` is
    not about the run this command just watched — it is older than it."""
    directory, _ = project
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    _write_marker(directory, "some-older-run", _seconds_ago(3600))

    def fake_run_crew() -> None:
        record_last_run(directory, "the-new-run", recorded_at=_now())

    monkeypatch.setattr(run_crew_module, "run_crew", fake_run_crew)
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.calls[0] == ("create", "the-new-run")


def test_the_app_saying_it_had_nothing_to_grade_is_not_a_run_to_grade(
    project, monkeypatch, capsys
):
    """An app that looked and found no trace has answered; the command says so
    once — the screen that showed it is gone — and grades nothing."""
    directory, _ = project
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)

    def fake_run_crew() -> None:
        # another run in the same project finishes while this app is open
        record_last_run(directory, "somebody-elses-run")
        eval_module.record_evaluation_outcome(None)  # what the app says

    monkeypatch.setattr(run_crew_module, "run_crew", fake_run_crew)
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    amp = install(monkeypatch, FakeAMP())

    with pytest.raises(SystemExit) as exit_code:
        eval_module.eval_crew()

    assert exit_code.value.code == 1
    assert amp.calls == []  # never somebody else's run
    assert "no trace was recorded" in capsys.readouterr().out


def test_a_concurrent_run_cannot_send_the_command_after_the_wrong_one(
    project, monkeypatch
):
    """The record is the project's last finish, which another run can replace
    while this app is open. The app's own marker is what settles it."""
    directory, _ = project
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)

    def fake_run_crew() -> None:
        eval_module.record_evaluation_outcome("the-run-the-app-watched")
        record_last_run(directory, "a-run-that-finished-later")

    monkeypatch.setattr(run_crew_module, "run_crew", fake_run_crew)
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.calls == []  # the app graded its own run; nothing is graded twice


def test_a_run_that_recorded_nothing_says_so(project, monkeypatch, capsys):
    """Nothing was traced, so nothing was evaluated and the app had nothing to
    show: this is the one thing the command still has to say."""
    directory, _ = project
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.click, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(run_crew_module, "run_crew", lambda: None)
    monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)
    amp = install(monkeypatch, FakeAMP())

    with pytest.raises(SystemExit) as exit_code:
        eval_module.eval_crew()

    assert exit_code.value.code == 1
    assert amp.calls == []
    assert "no trace was recorded" in capsys.readouterr().out


def test_a_run_from_hours_ago_is_not_waited_for(project, monkeypatch, capsys):
    """Nothing is in flight: AMP does not hold this run, and saying so at once
    is the useful answer."""
    directory, _ = project
    record_last_run(directory, recorded_at=_seconds_ago(7200))
    amp = install(monkeypatch, FakeAMP(create=[NOT_FOUND, QUEUED]))

    with pytest.raises(SystemExit):
        eval_module.eval_crew()

    assert [call for call in amp.calls if call[0] == "create"] == [
        ("create", EXECUTION_ID)
    ]
    assert "No spans recorded" in capsys.readouterr().out


def test_an_id_somebody_typed_is_answered_at_once(project, monkeypatch):
    """`--run <id>` is a question about a named run, not about this project's
    last one, and a typo should not cost two minutes."""
    directory, _ = project
    record_last_run(directory, recorded_at=_seconds_ago(5))
    amp = install(monkeypatch, FakeAMP(create=[NOT_FOUND, QUEUED]))

    with pytest.raises(SystemExit):
        eval_module.eval_crew(run_id="1d6b3f70-0000-4000-8000-000000000000")

    assert [call for call in amp.calls if call[0] == "create"] == [
        ("create", "1d6b3f70-0000-4000-8000-000000000000")
    ]


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

    # It stops the evaluation as a VALUE, not as an exit: the same sentence has
    # to reach a terminal and a screen, and only one of them is a terminal.
    for broken in (OSError(13, "Permission denied"), ValueError("Fernet key must be 32 url-safe base64-encoded bytes.")):
        monkeypatch.setattr(eval_module, "get_auth_token", lambda error=broken: (_ for _ in ()).throw(error))
        with pytest.raises(eval_module.EvaluationStoppedError) as stopped:
            eval_module.saved_login()
        said = str(stopped.value)
        assert "Could not read the saved login" in said and type(broken).__name__ in said and "crewai login" in said


def test_the_command_still_prints_that_sentence_and_exits(project, monkeypatch, capsys):
    """What the app shows on its screen, the command says in the terminal."""
    directory, _ = project
    record_last_run(directory)
    stopped = eval_module.EvaluationStoppedError(
        "Could not read the saved login (OSError: Permission denied). "
        "Run `crewai login` again, or `crewai eval` will not know who you are."
    )
    monkeypatch.setattr(
        eval_module, "saved_login", lambda: (_ for _ in ()).throw(stopped)
    )
    amp = install(monkeypatch, FakeAMP())

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    assert exit_.value.code == 1
    assert amp.calls == []  # stopped before anything was asked of AMP
    assert "Could not read the saved login" in capsys.readouterr().out


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


@pytest.mark.parametrize(
    "tracing, login, says_login",
    [
        ("true", None, True),  # tracing on, nobody logged in: log in
        ("true", "tok", False),  # logged in: the ordinary steps
        (None, None, False),  # tracing off: turn it on
    ],
)
def test_an_unattended_run_with_nothing_traced_says_what_would_trace_it(
    project, monkeypatch, capsys, tracing, login, says_login
):
    """With no terminal, an anonymous run's trace stays on the machine even with
    tracing on. Telling that user to turn tracing on sends them round the same
    loop; logging in is what makes an unattended run traced."""
    directory, _ = project
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(eval_module, "saved_login", lambda: login)
    if tracing:
        monkeypatch.setenv("CREWAI_TRACING_ENABLED", tracing)
    else:
        monkeypatch.delenv("CREWAI_TRACING_ENABLED", raising=False)

    with pytest.raises(SystemExit) as exited:
        eval_module.eval_crew()

    out = capsys.readouterr().out.replace("\n", " ")
    assert exited.value.code == 1
    assert ("run `crewai login`" in out) is says_login
    assert ("add CREWAI_TRACING_ENABLED=true" in out) is not says_login


def test_an_unreadable_login_is_the_reason_given_when_nothing_was_traced(
    project, monkeypatch, capsys
):
    """A login that exists and cannot be read is why an unattended run was not
    traced, and its own sentence says what to do — not "turn tracing on"."""
    directory, _ = project
    (directory / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: False)
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")

    def unreadable() -> None:
        raise eval_module.EvaluationStoppedError(
            "Could not read the saved login (PermissionError: [Errno 13] "
            "Permission denied: [/Users/me/.config/crewai]). Run `crewai login` again"
        )

    monkeypatch.setattr(eval_module, "saved_login", unreadable)

    with pytest.raises(SystemExit):
        eval_module.eval_crew()

    out = capsys.readouterr().out.replace("\n", " ")
    assert "Could not read the saved login" in out
    # printed as it is: `[/Users/…]` read as markup is a closing tag, and a crash
    assert "[/Users/me/.config/crewai]" in out
    assert "add CREWAI_TRACING_ENABLED=true" not in out


# ── Mode 1 files the evaluation under the project ───────────────────────────


def test_the_evaluation_is_filed_under_the_project(project, monkeypatch):
    directory, _ = project
    record_last_run(directory)
    monkeypatch.setattr(eval_module, "get_or_create_project_id", lambda: "proj-1")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.eval_crew()

    assert amp.sent_project_id == "proj-1"


def test_the_app_files_its_evaluation_under_the_project_too(project, monkeypatch):
    """The run app's door reads the id without minting one: it runs inside a
    crew's process, where rewriting pyproject.toml is not its business."""
    monkeypatch.setattr(eval_module, "get_project_id", lambda: "proj-2")
    amp = install(monkeypatch, FakeAMP(statuses=[done()]))

    eval_module.evaluate_run(EXECUTION_ID, on_started=lambda started: None)

    assert amp.sent_project_id == "proj-2"


# ── crewai eval --models ─────────────────────────────────────────────────────

MODELS = "openai/gpt-4o-mini, openrouter/meta-llama/llama-4-maverick"
DEPLOYMENT = "9b0a4a53-3f7c-4a8e-9a55-2f1c1e0f8e11"


def comparison(**overrides):
    body = {
        "models": [
            {
                "key": "base", "label": "Poem composer: openai/gpt-5.6-sol", "baseline": True,
                "grades": {"goal": 5, "tasks": 4, "agents": 3, "tools": None},
                "gate": "passed", "cost_usd": 0.0040, "seconds": 9.5, "tokens": 900,
            },
            {
                "key": "mini", "label": "Poem composer: openai/gpt-4o-mini", "baseline": False,
                "grades": {"goal": 5, "tasks": 5, "agents": 2, "tools": None},
                "gate": "passed", "cost_usd": 0.0012, "seconds": 7.6, "tokens": 736,
            },
        ],
        "suggestions": [
            {"subject": "agent:Poem composer", "field": "backstory", "problem": "only one model had it",
             "evidence": "x", "change": "only-one change", "models_affected": ["mini"], "shared": False},
            {"subject": "agent:Poem composer", "field": "goal", "problem": "the poem ignores the topic",
             "evidence": "y", "change": "Write a poem about {topic}.", "models_affected": ["base", "mini"],
             "shared": True},
            {"subject": "task:write", "field": "expected_output", "problem": "no length", "change": "4 lines"},
            {"subject": "agent:Poem composer", "field": "tools", "problem": "a fourth", "change": "never shown"},
        ],
    }
    body.update(overrides)
    return body


def compared(**overrides):
    return httpx.Response(200, json={"id": "ev-9", "status": "done", "url": URL, "comparison": comparison(**overrides)})


class FakeModelsAMP(FakeAMP):
    def __init__(self, create=None, statuses=None):
        super().__init__(create=create if create is not None else httpx.Response(
            202, json={"id": "ev-9", "url": URL, "status": "queued"}
        ), statuses=statuses)
        self.sent: dict | None = None

    def create_models_evaluation(self, models, *, project_id, eval_config=None, deployment_id=None):
        self.calls.append(("create_models", tuple(models)))
        self.sent = {"models": models, "project_id": project_id, "eval_config": eval_config,
                     "deployment_id": deployment_id}
        return self.create


@pytest.fixture
def deployed(project, monkeypatch):
    directory, opened = project
    (directory / "pyproject.toml").write_text('[tool.crewai]\nproject_id = "proj-1"\n')
    monkeypatch.setattr(eval_module, "get_or_create_project_id", lambda: "proj-1")
    return directory, opened


def test_models_are_compared_on_the_deployment_and_the_table_printed(deployed, monkeypatch, capsys):
    directory, opened = deployed
    (directory / "eval.jsonc").write_text('{"dataset": []}')
    running = httpx.Response(200, json={"id": "ev-9", "status": "running", "progress": {
        "event": "configuration_started", "payload": {"index": 1, "total": 3, "key": "mini"}}})
    judging = httpx.Response(200, json={"id": "ev-9", "status": "running", "progress": {
        "event": "judging", "payload": {"subject": "agent 'Poem composer'", "index": 1, "total": 3}}})
    amp = install(monkeypatch, FakeModelsAMP(statuses=[running, running, judging, compared()]))

    eval_module.eval_models(MODELS)

    out = capsys.readouterr().out
    assert amp.api_key == "login-token"
    assert amp.sent == {
        "models": ["openai/gpt-4o-mini", "openrouter/meta-llama/llama-4-maverick"],
        "project_id": "proj-1", "eval_config": '{"dataset": []}', "deployment_id": None,
    }
    assert opened == [URL] and URL in out
    # Progress: said once per change, never once per poll.
    assert out.count("running · model 2 of 3 · mini") == 1
    assert "judging agent 'Poem composer' · model 2 of 3" in out
    lines = out.splitlines()
    base = next(line for line in lines if "gpt-5.6-sol" in line)
    mini = next(line for line in lines if "Poem composer: openai/gpt-4o-mini" in line)
    assert "(deployed)" in base and "(deployed)" not in mini
    # The best per column: tasks and agents differ, goal is a tie, tools nothing.
    assert "5/5 ★" in mini and "4/5" in base and "4/5 ★" not in base
    assert "3/5 ★" in base and "2/5 ★" not in mini
    assert "$0.0012 ★" in mini and "7.6s ★" in mini and "9.5s ★" not in base
    assert "5/5 ★" not in base  # goal: both 5, nothing to point at
    # The top three suggestions, the one every model needed first.
    assert "What would make it better" in out
    assert out.index("the poem ignores the topic") < out.index("only one model had it")
    assert "change: Write a poem about {topic}." in out
    assert "a fourth" not in out


def test_the_comparison_is_counted_with_the_models_and_nothing_that_names_it(deployed, monkeypatch):
    spans: list[tuple[str, dict[str, str]]] = []

    class FakeTelemetry:
        def set_tracer(self) -> None:
            pass

        def feature_usage_span(self, feature, attributes=None) -> None:
            spans.append((feature, attributes or {}))

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", FakeTelemetry)
    install(monkeypatch, FakeModelsAMP(statuses=[compared()]))

    eval_module.eval_models(MODELS)

    assert spans == [("cli_usage:eval_models", {
        "authenticated": "true",
        # A public model by name; one crewAI's catalog does not list, by provider.
        "models": "openai/gpt-4o-mini,openrouter/other",
        "models_count": "2",
    })]


def test_a_refused_comparison_is_not_counted(deployed, monkeypatch, capsys):
    spans: list[str] = []

    class FakeTelemetry:
        def set_tracer(self) -> None:
            pass

        def feature_usage_span(self, feature, attributes=None) -> None:
            spans.append(feature)

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", FakeTelemetry)
    refused = httpx.Response(404, json={"error": "deployment_not_found", "message": (
        "No deployment of this project; deploy it with `crewai deploy create`, or pass --deployment.")})
    install(monkeypatch, FakeModelsAMP(create=refused))

    with pytest.raises(SystemExit) as stopped:
        eval_module.eval_models(MODELS)

    assert stopped.value.code == 1 and spans == []
    assert "deploy it with `crewai deploy create`, or pass --deployment." in capsys.readouterr().out


def test_the_deployment_named_is_sent(deployed, monkeypatch):
    amp = install(monkeypatch, FakeModelsAMP(statuses=[compared()]))

    eval_module.eval_models("openai/gpt-4o-mini", deployment_id=f" {DEPLOYMENT.upper()} ")

    assert amp.sent["deployment_id"] == DEPLOYMENT


def test_a_deployment_that_is_not_a_uuid_is_refused_before_anything_is_sent(deployed, monkeypatch, capsys):
    amp = install(monkeypatch, FakeModelsAMP())

    with pytest.raises(SystemExit) as stopped:
        eval_module.eval_models("openai/gpt-4o-mini", deployment_id="poem_creator")

    assert stopped.value.code == 1 and amp.calls == []
    assert "is not a deployment id" in capsys.readouterr().out


def test_comparing_models_needs_the_login(deployed, monkeypatch, capsys):
    monkeypatch.setattr(eval_module, "saved_login", lambda: None)
    amp = install(monkeypatch, FakeModelsAMP())

    with pytest.raises(SystemExit) as stopped:
        eval_module.eval_models(MODELS)

    assert stopped.value.code == 1 and amp.calls == []
    assert "log in with `crewai login`" in capsys.readouterr().out


def test_comparing_models_happens_from_the_project(project, monkeypatch, capsys):
    amp = install(monkeypatch, FakeModelsAMP())

    with pytest.raises(SystemExit) as stopped:
        eval_module.eval_models(MODELS)

    assert stopped.value.code == 1 and amp.calls == []
    assert "No crewAI project here" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("openai/gpt-4o-mini", ["openai/gpt-4o-mini"]),
        (" openai/gpt-4o-mini , openai/gpt-4o-mini,anthropic/claude-haiku-4-5, ",
         ["openai/gpt-4o-mini", "anthropic/claude-haiku-4-5"]),
        ("openrouter/meta-llama/llama-4-maverick", ["openrouter/meta-llama/llama-4-maverick"]),
    ],
)
def test_the_models_are_one_list_stripped_with_repeats_dropped(text, expected):
    assert eval_module.parse_models(text) == expected


@pytest.mark.parametrize(
    ("text", "said"),
    [
        ("gpt-4o-mini", "names no provider"),
        ("openai/", "names no provider"),
        ("/gpt-4o", "names no provider"),
        (" , ", "names no model"),
        (",".join(f"openai/m{n}" for n in range(6)), "at most 5"),
        ("openai/" + "m" * 200, "longer than 200"),
    ],
)
def test_a_list_amp_would_refuse_is_refused_here_with_an_example(text, said):
    with pytest.raises(eval_module.EvaluationStoppedError, match=said) as refused:
        eval_module.parse_models(text)
    if "provider" in said or "names no model" in said:
        assert "openai/gpt-4o-mini" in str(refused.value)


def test_a_model_without_its_provider_exits_one_before_anything_is_sent(deployed, monkeypatch, capsys):
    amp = install(monkeypatch, FakeModelsAMP())

    with pytest.raises(SystemExit) as stopped:
        eval_module.eval_models("gpt-4o-mini")

    assert stopped.value.code == 1 and amp.calls == []
    assert "provider/model" in capsys.readouterr().out


def test_a_failed_comparison_exits_one_with_the_reason(deployed, monkeypatch, capsys):
    failed = httpx.Response(200, json={"id": "ev-9", "status": "failed", "error": "the deployment answered 500"})
    install(monkeypatch, FakeModelsAMP(statuses=[failed]))

    with pytest.raises(SystemExit) as stopped:
        eval_module.eval_models(MODELS)

    assert stopped.value.code == 1
    assert "Comparison failed: the deployment answered 500" in capsys.readouterr().out


def test_done_without_a_comparison_is_a_protocol_error(deployed, monkeypatch, capsys):
    install(monkeypatch, FakeModelsAMP(statuses=[httpx.Response(200, json={"id": "ev-9", "status": "done"})]))

    with pytest.raises(SystemExit) as stopped:
        eval_module.eval_models(MODELS)

    assert stopped.value.code == 1
    assert "done without a comparison (protocol error)" in capsys.readouterr().out


def test_a_deployment_this_account_may_not_run_is_amps_sentence_alone(deployed, monkeypatch, capsys):
    """Logging in again changes nothing about a deployment one may not run."""
    refused = httpx.Response(403, json={"error": "not_allowed", "message": "You may not run poem_creator."})
    install(monkeypatch, FakeModelsAMP(create=refused))

    with pytest.raises(SystemExit):
        eval_module.eval_models(MODELS)

    out = capsys.readouterr().out
    assert "You may not run poem_creator." in out and "crewai login" not in out


def test_everything_from_the_wire_prints_literally(deployed, monkeypatch, capsys):
    hostile = comparison(
        models=[{"key": "k", "label": "[red]Writer[/red]: [link=https://evil.test]x[/link]", "baseline": True,
                 "grades": {"goal": 9, "tasks": True, "agents": "5", "tools": None}, "cost_usd": True,
                 "seconds": -1}],
        suggestions=[{"subject": "[bold]agent[/bold]", "field": "goal", "problem": "[red]p[/red]",
                      "change": "[link=https://evil.test]c[/link]"}],
    )
    install(monkeypatch, FakeModelsAMP(statuses=[httpx.Response(
        200, json={"id": "ev-9", "status": "done", "url": URL, "comparison": hostile})]))

    eval_module.eval_models(MODELS)

    out = capsys.readouterr().out
    assert "[red]Writer[/red]: [link=https://evil.test]x[/link]" in out
    assert "[red]p[/red]" in out and "[link=https://evil.test]c[/link]" in out
    # Not a grade, a cost or a time: each prints as "—", never as a value.
    row = next(line for line in out.splitlines() if "Writer" in line)
    assert "9/5" not in row and "True" not in row and "-1" not in row and "5/5" not in row
    assert row.count("—") == 6


@pytest.mark.parametrize(
    ("progress", "line"),
    [
        ("Setting up the deployment", "Setting up the deployment"),
        ({"message": "model 2 of 3: mini"}, "model 2 of 3: mini"),
        ({"event": "configuration_done", "payload": {"index": 0, "total": 2, "key": "base"}},
         "graded · model 1 of 2 · base"),
        ({"subject": "the final output"}, "judging the final output"),
        ({"unknown": "shape"}, None),
        (["not", "a", "dict"], None),
    ],
)
def test_the_progress_line_says_what_is_there_and_guesses_nothing(progress, line):
    assert eval_module._progress_line({"progress": progress}) == line


def test_the_last_event_stands_in_for_progress():
    events = [{"event": "configuration_started", "payload": {"index": 0, "total": 2, "key": "base"}},
              {"event": "judging", "payload": {"subject": "task 'write'"}}]

    assert eval_module._progress_line({"events": events}) == "judging task 'write'"


def test_the_cli_maps_models_and_deployment_to_the_comparison(monkeypatch):
    calls = []
    monkeypatch.setattr("crewai_cli.cli.eval_models", lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr("crewai_cli.cli.eval_crew", lambda **kwargs: calls.append(("run", kwargs)))
    runner = CliRunner()

    assert runner.invoke(eval_command, ["--models", MODELS]).exit_code == 0
    assert runner.invoke(eval_command, ["--models", "openai/gpt-4o", "--deployment", DEPLOYMENT]).exit_code == 0
    assert calls == [((MODELS,), {"deployment_id": None}), (("openai/gpt-4o",), {"deployment_id": DEPLOYMENT})]

    both = runner.invoke(eval_command, ["--models", MODELS, "--run", EXECUTION_ID])
    alone = runner.invoke(eval_command, ["--deployment", DEPLOYMENT])
    assert both.exit_code == 2 and "Give one of them" in both.output
    assert alone.exit_code == 2 and "add --models LIST" in alone.output
    assert len(calls) == 2
    assert "--models" in runner.invoke(eval_command, ["--help"]).output


def test_a_comparison_leaves_its_criteria_behind_when_the_project_has_none(deployed, monkeypatch, capsys):
    directory, _ = deployed
    answer = compared()
    body = {**answer.json(), "eval_config": '// the criteria\n{"dataset": []}\n'}
    install(monkeypatch, FakeModelsAMP(statuses=[httpx.Response(200, json=body)]))

    eval_module.eval_models(MODELS)

    assert (directory / "eval.jsonc").read_text() == '// the criteria\n{"dataset": []}\n'
    assert "Wrote eval.jsonc" in capsys.readouterr().out


def test_a_comparison_never_overwrites_the_projects_criteria(deployed, monkeypatch, capsys):
    directory, _ = deployed
    (directory / "eval.jsonc").write_text("// ours\n{}\n")
    body = {**compared().json(), "eval_config": "// theirs\n{}\n"}
    install(monkeypatch, FakeModelsAMP(statuses=[httpx.Response(200, json=body)]))

    eval_module.eval_models(MODELS)

    assert (directory / "eval.jsonc").read_text() == "// ours\n{}\n"
    assert "Wrote eval.jsonc" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("typed", "sent"),
    [
        ("openai/gpt-4o-mini", "openai/gpt-4o-mini"),
        ("anthropic/claude-haiku-4-5", "anthropic/claude-haiku-4-5"),
        ("openrouter/openai/gpt-4o-mini", "openrouter/openai/gpt-4o-mini"),
        # A customer's own: a fine-tune, a deployment name, a model named after
        # a public one, an unknown name — by provider only.
        ("openai/ft:gpt-4o-mini:acme-corp::abc", "openai/other"),
        ("azure/my-deployment", "azure/other"),
        ("openai/gpt-4o-acme", "openai/other"),
        ("ollama/acme-llama", "ollama/other"),
        # A provider crewAI does not know can name a host.
        ("llm.acme.internal/llama-3", "other/other"),
    ],
)
def test_usage_stats_name_a_model_only_when_it_is_a_public_one(typed, sent):
    assert eval_module.telemetry_model_name(typed) == sent


def test_without_crewais_catalog_every_model_is_other(monkeypatch):
    monkeypatch.setattr(eval_module, "_known_models_and_providers", lambda: (frozenset(), frozenset()))

    assert eval_module.telemetry_model_name("openai/gpt-4o-mini") == "other/other"


# ── the brief for a coding agent ─────────────────────────────────────────────

BRIEF = (
    "# What failed\n\n"
    "1. task write — [Errno 13] the poem ignores [red]the topic[/red]\n"
    "   change: set `expected_output` to \"Four lines about {topic}.\"\n"
)
BRIEF_URL = "https://evolve.crewai.test/e/ev-1.md"


def _done_with(**fields):
    return httpx.Response(200, json={**done(gate="failed").json(), **fields})


def test_with_nobody_watching_the_brief_follows_the_verdict_and_the_link_literally(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory)
    monkeypatch.setattr(eval_module, "nobody_watching", lambda: True)
    install(monkeypatch, FakeAMP(statuses=[_done_with(brief_markdown=BRIEF, brief_url=BRIEF_URL)]))

    with pytest.raises(SystemExit) as exit_:
        eval_module.eval_crew()

    out = capsys.readouterr().out
    assert exit_.value.code == 1  # the gate's, unchanged
    assert BRIEF in out  # brackets and all: never read as markup
    assert out.index("Goal gate: FAILED") < out.index(f"Full report: {URL}") < out.index(BRIEF)
    assert "Brief (markdown)" not in out


def test_with_only_the_briefs_url_the_link_is_printed(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory)
    monkeypatch.setattr(eval_module, "nobody_watching", lambda: True)
    install(monkeypatch, FakeAMP(statuses=[_done_with(brief_url=BRIEF_URL)]))

    with pytest.raises(SystemExit):
        eval_module.eval_crew()

    assert f"Brief (markdown): {BRIEF_URL}\n" in capsys.readouterr().out


@pytest.mark.parametrize("brief_url", [None, "javascript:alert(1)"])
def test_an_older_server_adds_nothing_after_the_link(project, monkeypatch, capsys, brief_url):
    directory, _ = project
    record_last_run(directory)
    monkeypatch.setattr(eval_module, "nobody_watching", lambda: True)
    fields = {"brief_url": brief_url} if brief_url else {}
    install(monkeypatch, FakeAMP(statuses=[_done_with(**fields)]))

    with pytest.raises(SystemExit):
        eval_module.eval_crew()

    out = capsys.readouterr().out
    assert out.endswith(f"Full report: {URL}\n") and "Brief" not in out


def test_in_a_terminal_the_brief_is_not_printed(project, monkeypatch, capsys):
    directory, _ = project
    record_last_run(directory)
    monkeypatch.setattr(eval_module, "nobody_watching", lambda: False)
    install(monkeypatch, FakeAMP(statuses=[_done_with(brief_markdown=BRIEF, brief_url=BRIEF_URL)]))

    with pytest.raises(SystemExit):
        eval_module.eval_crew()

    out = capsys.readouterr().out
    assert "What failed" not in out and "Brief" not in out
    assert out.endswith(f"Full report: {URL}\n")


def test_nobody_is_watching_when_stdout_is_not_a_terminal(monkeypatch):
    monkeypatch.setattr(eval_module, "is_dmn_mode_enabled", lambda: False)
    monkeypatch.setattr(eval_module.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    assert eval_module.nobody_watching() is False
    monkeypatch.setattr(eval_module.sys, "stdout", SimpleNamespace(isatty=lambda: False))
    assert eval_module.nobody_watching() is True
    monkeypatch.setattr(eval_module.sys, "stdout", SimpleNamespace())  # no isatty at all
    assert eval_module.nobody_watching() is True
    monkeypatch.setattr(eval_module.sys, "stdout", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(eval_module, "is_dmn_mode_enabled", lambda: True)
    assert eval_module.nobody_watching() is True


@pytest.mark.parametrize("watching", [False, True])
def test_a_comparison_prints_its_brief_only_when_nobody_is_watching(deployed, monkeypatch, capsys, watching):
    monkeypatch.setattr(eval_module, "nobody_watching", lambda: not watching)
    body = {**compared().json(), "brief_markdown": BRIEF, "brief_url": BRIEF_URL}
    install(monkeypatch, FakeModelsAMP(statuses=[httpx.Response(200, json=body)]))

    eval_module.eval_models(MODELS)  # a finished comparison exits 0, as before

    out = capsys.readouterr().out
    if watching:
        assert "What failed" not in out and out.endswith(f"Full report: {URL}\n")
    else:
        assert out.index(f"Full report: {URL}") < out.index(BRIEF)


def test_the_help_tells_an_agent_it_gets_a_brief():
    help_text = " ".join(CliRunner().invoke(eval_command, ["--help"]).output.split())
    assert (
        "Run by a script or coding agent (no terminal), it prints a markdown brief "
        "after the verdict — what failed and the change to make — so an agent can act on it directly"
    ) in help_text
    assert "it prints a markdown brief after the comparison" in help_text
