"""Evaluate's sharing question: asked only of a person who is not logged in.

Mirrors the consent cases of `lib/cli/tests/experimental/test_eval_crew.py` and
`lib/cli/tests/test_crew_run_tui.py` against `crewai_cli` internals.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from crewai_cli import crew_run_tui, run_crew as run_crew_module
from crewai_cli.crew_run_tui import CrewRunApp
from crewai_cli.experimental import eval_crew as eval_module
from crewai_core.token_manager import TokenManager


_SHIPPED_SAVED_LOGIN = eval_module.saved_login


class _FakeAMP:
    """A PlusAPI double that starts an evaluation and answers it passed."""

    def __init__(self) -> None:
        self.created: list[str] = []
        self.api_key: str | None = None
        self.base_url = "https://amp.test"
        self.headers: dict[str, str] = {}
        # What AMP says of the saved login, and which login each check carried.
        self.login_answer: httpx.Response | Exception = httpx.Response(200, json=[])
        self.login_checks: list[str | None] = []

    def get_organizations(self):
        self.login_checks.append(self.api_key)
        if isinstance(self.login_answer, Exception):
            raise self.login_answer
        return self.login_answer

    def create_evaluation(self, execution_id, *, eval_config=None, project_id=None):
        self.created.append(execution_id)
        return httpx.Response(202, json={"id": "ev-1", "url": "https://amp.test/e/1"})

    def get_evaluation(self, evaluation_id):
        return httpx.Response(
            200,
            json={"id": evaluation_id, "status": "done", "verdict": {"gate": "passed", "grades": {}}},
        )


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project that never traced, at a terminal, logged in to https://amp.test
    (the AMP this machine is configured for). Returns what the test reads back."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    for name in ("CREWAI_TRACING_ENABLED", "CREWAI_PLUS_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        "crewai_core.user_data.is_tracing_enabled",
        lambda: os.environ.get("CREWAI_TRACING_ENABLED") == "true",
    )
    for where in ("crewai_cli.crew_run_tui._awaiting_dir", "crewai_core.eval_handshake.awaiting_dir"):
        monkeypatch.setattr(where, lambda: tmp_path / ".eval-awaiting")
    monkeypatch.setattr(eval_module, "_EVAL_ID", ContextVar("crewai_eval_id", default=None))
    monkeypatch.setattr(eval_module, "get_or_create_project_id", lambda: None)
    monkeypatch.setattr(eval_module, "saved_login", lambda: "login-token")
    # The `crewai login` store and the account's own are one folder here: a
    # saved login counts as this account's.
    store = tmp_path / ".credentials"
    store.mkdir()
    monkeypatch.setattr(TokenManager, "_get_secure_storage_path", staticmethod(lambda: store))
    monkeypatch.setattr(eval_module, "_account_credentials_dir", lambda: store)
    monkeypatch.setattr(eval_module, "is_dmn_mode_enabled", lambda: False)
    monkeypatch.setattr(eval_module, "_open", lambda url: None)
    monkeypatch.setattr(eval_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(eval_module.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        eval_module, "Settings", lambda: SimpleNamespace(enterprise_base_url="https://amp.test")
    )
    amp = _FakeAMP()

    def plus_api(api_key=None, base_url=None):
        amp.api_key = api_key
        amp.base_url = base_url or os.environ.get("CREWAI_PLUS_URL") or "https://amp.test"
        return amp

    monkeypatch.setattr(eval_module, "PlusAPI", plus_api)

    ran: list[tuple[str, object]] = []

    def fake_run_crew() -> None:
        from crewai_core.eval_handshake import awaited_evaluation

        said = awaited_evaluation() or {}
        ran.append((os.environ.get("CREWAI_TRACING_ENABLED", ""), said.get("tracing_turned_on")))
        (tmp_path / ".crewai").mkdir(exist_ok=True)
        (tmp_path / ".crewai" / "last_run.json").write_text(
            json.dumps({"execution_id": "the-new-run", "recorded_at": datetime.now(timezone.utc).isoformat()})
        )

    monkeypatch.setattr(run_crew_module, "run_crew", fake_run_crew)

    asked: list[str] = []
    answer = {"yes": True}

    def confirm(text, **kwargs):
        asked.append(text)
        return answer["yes"]

    monkeypatch.setattr(eval_module.click, "confirm", confirm)

    usage: list[str] = []

    class FakeTelemetry:
        def set_tracer(self) -> None:
            pass

        def feature_usage_span(self, feature, attributes=None) -> None:
            usage.append(feature)

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", FakeTelemetry)
    return SimpleNamespace(
        directory=tmp_path, amp=amp, ran=ran, asked=asked, answer=answer, usage=usage
    )


def _env(directory: Path) -> str:
    path = directory / ".env"
    return path.read_text() if path.exists() else ""


def test_logged_in_the_crew_runs_traced_and_one_line_says_so(project, capsys) -> None:
    eval_module.eval_crew()
    out = " ".join(capsys.readouterr().out.split())

    assert project.asked == []
    assert project.ran == [("true", True)]
    assert "CREWAI_TRACING_ENABLED=true" in _env(project.directory)
    assert eval_module.TRACING_NOW_ON in out
    assert project.amp.created == ["the-new-run"]
    assert project.amp.login_checks == ["login-token"]
    assert project.usage[0] == "cli_usage:eval_consent:logged_in"


@pytest.mark.parametrize(
    "login_answer",
    [
        httpx.Response(401, json={"error": "Bad credentials - User not found"}),
        httpx.ConnectError("unreachable"),
    ],
    ids=["401 user not found", "unreachable"],
)
def test_a_login_amp_refuses_is_asked_as_logged_out(project, capsys, login_answer) -> None:
    """Its trace would go up anonymously, not to the login's organization."""
    project.amp.login_answer = login_answer
    project.answer["yes"] = False

    with pytest.raises(SystemExit) as exited:
        eval_module.eval_crew()
    out = " ".join(capsys.readouterr().out.split())

    assert exited.value.code == 0
    assert project.asked == ["Get a score for this run?"]
    assert eval_module.TRACING_NOW_ON not in out
    assert project.ran == [] and project.amp.created == []
    assert _env(project.directory) == ""
    assert project.usage == ["cli_usage:eval_stopped:declined"]


@pytest.mark.parametrize("yes", [True, False])
def test_logged_out_the_person_is_asked_and_only_a_no_runs_nothing(
    project, monkeypatch, capsys, yes
) -> None:
    monkeypatch.setattr(eval_module, "saved_login", lambda: None)
    project.answer["yes"] = yes

    if yes:
        eval_module.eval_crew()
    else:
        with pytest.raises(SystemExit) as exited:
            eval_module.eval_crew()
        assert exited.value.code == 0
    out = " ".join(capsys.readouterr().out.split())

    assert project.asked == ["Get a score for this run?"]
    assert "CrewAI AMP grades it, then shows you what to fix." in out
    assert "The run's trace is sent to CrewAI AMP." in out
    if yes:
        assert project.ran == [("true", False)]
        assert project.amp.created == ["the-new-run"]
    else:
        assert project.ran == [] and project.amp.created == []
        assert _env(project.directory) == ""
        assert project.usage == ["cli_usage:eval_stopped:declined"]


def test_logged_in_a_project_that_turned_tracing_off_is_still_asked(project) -> None:
    (project.directory / ".env").write_text("CREWAI_TRACING_ENABLED=false\n")
    project.answer["yes"] = False

    with pytest.raises(SystemExit):
        eval_module.eval_crew()

    assert project.asked == [eval_module.TRACING_OFF_QUESTION]
    assert project.ran == []
    assert project.usage == ["cli_usage:eval_stopped:declined"]


def test_logged_in_toward_an_untrusted_amp_nothing_is_run_or_asked(project) -> None:
    (project.directory / ".env").write_text("CREWAI_PLUS_URL=https://evil.example\n")

    with pytest.raises(SystemExit, match="1"):
        eval_module.eval_crew()

    assert project.asked == [] and project.ran == []
    assert project.usage == ["cli_usage:eval_stopped:untrusted_amp"]


@pytest.fixture
def real_saved_login(monkeypatch):
    """`saved_login` as shipped, over the project's login store in this test's folder."""
    monkeypatch.setattr(eval_module, "saved_login", _SHIPPED_SAVED_LOGIN)


@pytest.mark.parametrize(("expires_in", "asked"), [(-60, True), (3600, False)], ids=["expired", "live"])
def test_an_expired_login_is_treated_as_logged_out(
    project, monkeypatch, real_saved_login, expires_in, asked
) -> None:
    TokenManager().save_tokens("saved-token", int(time.time() + expires_in))

    eval_module.eval_crew()

    assert project.asked == (["Get a score for this run?"] if asked else [])
    assert project.ran == [("true", not asked)]


@pytest.mark.parametrize("logged_in", [True, False])
def test_the_run_apps_evaluate_asks_only_when_logged_out(project, monkeypatch, logged_in) -> None:
    if not logged_in:
        monkeypatch.setattr(eval_module, "saved_login", lambda: None)
    app = CrewRunApp()
    app._status = "completed"
    app._crew_result = object()
    app._telemetry = Mock()
    app._execution_uuid = "run-this-app"
    app._unsubscribe = lambda: None  # type: ignore[method-assign]
    exits: list[object] = []
    app.exit = lambda result=None: exits.append(result)  # type: ignore[method-assign]
    pushed: list[object] = []
    app.push_screen = lambda screen, callback=None: pushed.append(screen)  # type: ignore[method-assign]
    monkeypatch.setattr(crew_run_tui, "_trace_was_recorded", lambda uuid: False)

    app.action_evaluate_crew()

    if logged_in:
        assert pushed == [] and exits == [app._crew_result]
        # The app turns nothing on: `crewai eval` does, and says so.
        assert _env(project.directory) == ""
        assert os.environ.get("CREWAI_TRACING_ENABLED") != "true"
    else:
        assert [type(screen) for screen in pushed] == [crew_run_tui.EvaluateConsentScreen]
        assert exits == []
        assert (
            crew_run_tui.EvaluateConsentScreen.YES_LABEL,
            crew_run_tui.EvaluateConsentScreen.NO_LABEL,
        ) == ("Get score", "Not now")


def test_the_run_apps_evaluate_logged_in_hands_crewai_eval_the_line_to_say(
    project, monkeypatch, capsys
) -> None:
    """Pressed logged in, Evaluate closes the app and `crewai eval` runs the
    crew again: that command turns tracing on, says so in one line, and tells
    the run app it opens to say so too."""
    app = CrewRunApp()
    app._status = "completed"
    app._crew_result = object()
    app._telemetry = Mock()
    app._execution_uuid = "run-this-app"
    app._unsubscribe = lambda: None  # type: ignore[method-assign]
    app.exit = lambda result=None: None  # type: ignore[method-assign]
    app.push_screen = lambda screen, callback=None: pytest.fail("nothing is asked")  # type: ignore[method-assign]
    monkeypatch.setattr(crew_run_tui, "_trace_was_recorded", lambda uuid: False)

    app.action_evaluate_crew()
    assert app._want_traced_rerun is True
    run_crew_module._chain_traced_rerun()  # what `crewai run` does once the app closed
    out = " ".join(capsys.readouterr().out.split())

    assert project.asked == []
    assert project.ran == [("true", True)]
    assert f"{eval_module.RUNNING_THE_CREW['run_again']} {eval_module.TRACING_NOW_ON}" in out
    assert project.amp.created == ["the-new-run"]


def test_the_run_apps_evaluate_with_a_login_amp_refuses_is_asked_in_the_terminal(
    project, monkeypatch, capsys
) -> None:
    """The app reads the login on this machine only; `crewai eval` asks AMP
    before it skips the question, and asks the person when AMP refuses."""
    project.amp.login_answer = httpx.Response(401, json={"error": "Bad credentials - User not found"})
    project.answer["yes"] = False
    app = CrewRunApp()
    app._status = "completed"
    app._crew_result = object()
    app._telemetry = Mock()
    app._execution_uuid = "run-this-app"
    app._unsubscribe = lambda: None  # type: ignore[method-assign]
    app.exit = lambda result=None: None  # type: ignore[method-assign]
    monkeypatch.setattr(crew_run_tui, "_trace_was_recorded", lambda uuid: False)

    app.action_evaluate_crew()
    with pytest.raises(SystemExit) as exited:
        run_crew_module._chain_traced_rerun()

    assert exited.value.code == 0
    assert project.asked == ["Get a score for this run?"]
    assert project.ran == [] and _env(project.directory) == ""


# A cloned project's .env, moving the login store into a folder it ships.
_FOREIGN_HOME_ENV = "HOME=./.h\nXDG_DATA_HOME=./.h/.local/share\nLOCALAPPDATA=./.h/AppData/Local\n"
_SHIPPED_LOGIN_STORE = vars(TokenManager)["_get_secure_storage_path"]
_SHIPPED_ACCOUNT_CREDENTIALS_DIR = eval_module._account_credentials_dir


@pytest.fixture
def two_logins(project, tmp_path_factory, monkeypatch):
    """This account's home (account database and HOME) with a live login, and
    `.h` in the project, a home it ships, with someone else's live login. The
    stores, and the account's, are found as shipped."""
    monkeypatch.setattr(TokenManager, "_get_secure_storage_path", _SHIPPED_LOGIN_STORE)
    monkeypatch.setattr(eval_module, "_account_credentials_dir", _SHIPPED_ACCOUNT_CREDENTIALS_DIR)
    monkeypatch.setattr(eval_module, "saved_login", _SHIPPED_SAVED_LOGIN)
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setattr("pwd.getpwuid", lambda uid: SimpleNamespace(pw_dir=str(home)))
    # Put back after the test whatever the project's .env sets in os.environ.
    for name in ("XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    in_a_day = int(time.time() + 86400)
    monkeypatch.setenv("HOME", str(project.directory / ".h"))
    TokenManager().save_tokens("someone-elses-token", in_a_day)
    monkeypatch.setenv("HOME", str(home))
    TokenManager().save_tokens("this-accounts-token", in_a_day)


@pytest.mark.skipif(sys.platform == "win32", reason="the account database is POSIX's")
def test_a_login_the_projects_env_points_to_is_no_consent(project, two_logins, capsys) -> None:
    """The project's .env points HOME at its own valid login: the person is
    still asked, and a no runs nothing and asks AMP nothing."""
    (project.directory / ".env").write_text(_FOREIGN_HOME_ENV)
    project.answer["yes"] = False

    with pytest.raises(SystemExit) as exited:
        eval_module.eval_crew()
    out = " ".join(capsys.readouterr().out.split())

    assert exited.value.code == 0
    assert eval_module.saved_login() == "someone-elses-token"  # what the run would carry
    assert project.asked == ["Get a score for this run?"]
    assert eval_module.TRACING_NOW_ON not in out
    assert project.ran == [] and project.amp.created == [] and project.amp.login_checks == []
    assert _env(project.directory) == _FOREIGN_HOME_ENV
    assert project.usage == ["cli_usage:eval_stopped:declined"]


@pytest.mark.skipif(sys.platform == "win32", reason="the account database is POSIX's")
def test_this_accounts_own_login_still_skips_the_question(project, two_logins) -> None:
    eval_module.eval_crew()

    assert project.asked == []
    assert project.ran == [("true", True)]
    assert project.amp.login_checks == ["this-accounts-token"]


@pytest.mark.skipif(sys.platform == "win32", reason="the account database is POSIX's")
def test_the_run_apps_evaluate_asks_when_the_projects_env_points_to_a_login(
    project, two_logins, monkeypatch
) -> None:
    """The run already loaded the project's .env into this process."""
    monkeypatch.setenv("HOME", "./.h")
    monkeypatch.setenv("XDG_DATA_HOME", "./.h/.local/share")
    monkeypatch.setenv("LOCALAPPDATA", "./.h/AppData/Local")
    app = CrewRunApp()
    app._status = "completed"
    app._crew_result = object()
    app._telemetry = Mock()
    app._execution_uuid = "run-this-app"
    app._unsubscribe = lambda: None  # type: ignore[method-assign]
    exits: list[object] = []
    app.exit = lambda result=None: exits.append(result)  # type: ignore[method-assign]
    pushed: list[object] = []
    app.push_screen = lambda screen, callback=None: pushed.append(screen)  # type: ignore[method-assign]
    monkeypatch.setattr(crew_run_tui, "_trace_was_recorded", lambda uuid: False)

    app.action_evaluate_crew()

    assert [type(screen) for screen in pushed] == [crew_run_tui.EvaluateConsentScreen]
    assert exits == []
