"""Temporary AMP accounts: created when `crewai deploy` or `crewai eval --models`
needs a login and finds none, used for those requests, replaced by `crewai login`.

AMP is faked at the HTTP transport, so every request the CLI sends is real down
to the headers and nothing leaves the machine.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
from crewai_core.token_manager import TokenManager
import httpx
import pytest

from crewai_cli import run_crew
from crewai_cli.authentication import placeholder as placeholder_module
from crewai_cli.authentication.main import AuthenticationCommand
from crewai_cli.cli import deploy_create, deploy_push, logout
from crewai_cli.command import AuthenticationRequiredError
from crewai_cli.deploy.main import DeployCommand
from crewai_cli.experimental import eval_crew as eval_module
from crewai_cli.organization.main import OrganizationCommand


AMP = "https://amp.test"
PLACEHOLDERS = "/crewai_plus/api/v1/placeholder_accounts"
CREWS = "/crewai_plus/api/v1/crews"
PUSH = f"{CREWS}/by-name/demo/deploy"
STATUS = f"{CREWS}/by-name/demo/status"
EVALUATIONS = "/crewai_plus/api/v1/tracing/evaluations"
ORG = "0f3c2a5e-7d4b-4c1e-9a8f-1b2c3d4e5f60"
EXPIRES = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(days=7)
EXPIRY_DATE = EXPIRES.date().isoformat()
LOGIN_FIRST = "Please sign up/login to CrewAI+ before using the CLI."


def claim_url(token: str) -> str:
    return f"{AMP}/claim/code-for-{token}"


def created(token: str = "pat_first", **overrides: object) -> httpx.Response:
    body = {
        "token": token,
        "organization_uuid": ORG,
        "expires_at": EXPIRES.isoformat().replace("+00:00", "Z"),
        "claim_url": claim_url(token),
        **overrides,
    }
    return httpx.Response(201, json=body)


class FakeAMP:
    """AMP's routes this feature touches, answering from scripts; every request kept."""

    def __init__(
        self,
        *,
        placeholders: list[httpx.Response] | None = None,
        crews: list[int] | None = None,
        evaluations: list[int] | None = None,
        unreachable: bool = False,
    ) -> None:
        self.placeholders = list(placeholders or [])
        self.crews = list(crews or [])
        self.evaluations = list(evaluations or [])
        self.unreachable = unreachable
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.unreachable:
            raise httpx.ConnectError("connection refused", request=request)
        self.requests.append(request)
        path = request.url.path
        if path == PLACEHOLDERS:
            return self.placeholders.pop(0)
        if path in (CREWS, PUSH) and request.method == "POST":
            status = self.crews.pop(0)
            if status == 401:
                return httpx.Response(401, json={"error": "bad_credentials"})
            return httpx.Response(
                status, json={"uuid": "crew-uuid", "status": "Deploy Enqueued"}
            )
        if path == STATUS:
            return httpx.Response(404, json={"error": "not_found"})
        if path == EVALUATIONS and request.method == "POST":
            status = self.evaluations.pop(0)
            if status == 401:
                return httpx.Response(401, json={"error": "unauthorized"})
            return httpx.Response(
                202, json={"id": "ev-9", "url": None, "status": "queued"}
            )
        if path == f"{EVALUATIONS}/ev-9":
            return httpx.Response(200, json=FINISHED_COMPARISON)
        raise AssertionError(f"unexpected request {request.method} {request.url}")

    def to(self, path: str) -> list[httpx.Request]:
        return [request for request in self.requests if request.url.path == path]

    def bearers(self, path: str) -> list[str | None]:
        return [request.headers.get("Authorization") for request in self.to(path)]


FINISHED_COMPARISON = {
    "id": "ev-9",
    "status": "done",
    "comparison": {
        "models": [
            {
                "key": "base",
                "label": "Writer: openai/gpt-5",
                "baseline": True,
                "grades": {"goal": 5, "tasks": 4, "agents": 4, "tools": None},
                "gate": "passed",
            }
        ]
    },
}


@pytest.fixture
def credentials(tmp_path, monkeypatch) -> Path:
    """The saved login and the temporary account live here, not in the user's own store."""
    directory = tmp_path / "credentials"
    directory.mkdir()
    monkeypatch.setattr(
        TokenManager, "_get_secure_storage_path", staticmethod(lambda: directory)
    )
    return directory


@pytest.fixture
def amp(monkeypatch, credentials):
    """Install a FakeAMP behind every PlusAPI request (scripts are set per test)."""
    fake = FakeAMP()
    real_client = httpx.Client
    monkeypatch.setattr("crewai_core.plus_api._MACHINE_TRANSPORT_ENV", {})
    monkeypatch.setattr("crewai_core.plus_api._os_proxies", dict)
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(fake), **kwargs),
    )
    monkeypatch.setenv("CREWAI_PLUS_URL", AMP)
    return fake


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A deployable project with a git origin: create sends one POST to AMP."""
    directory = tmp_path / "demo"
    directory.mkdir()
    (directory / "pyproject.toml").write_text(
        '[project]\nname = "demo"\ndependencies = ["crewai"]\n\n[tool.crewai]\nproject_id = "proj-1"\n'
    )
    (directory / "uv.lock").write_text("# lock\n")
    monkeypatch.chdir(directory)
    origin = SimpleNamespace(origin_url=lambda: "https://github.com/acme/demo.git")
    monkeypatch.setattr(
        DeployCommand, "_prepare_git_repository", lambda self, **kwargs: origin
    )
    monkeypatch.setattr("crewai_cli.deploy.main.fetch_and_json_env_file", dict)
    monkeypatch.setattr("crewai_cli.deploy.main.webbrowser.open", lambda url: True)
    return directory


@pytest.fixture
def coding_agent(monkeypatch):
    monkeypatch.setenv("CLAUDECODE", "1")


@pytest.fixture
def at_a_terminal(monkeypatch):
    monkeypatch.setattr(
        placeholder_module.sys, "stdin", SimpleNamespace(isatty=lambda: True)
    )


def saved_token() -> str | None:
    saved = placeholder_module.saved_placeholder(AMP)
    return saved.token if saved else None


def save_placeholder(token: str, *, expires_at: datetime = EXPIRES, origin: str = AMP) -> None:
    TokenManager(file_path="placeholder.enc").save_tokens(
        json.dumps(
            {
                "token": token,
                "organization_uuid": ORG,
                "expires_at": expires_at.isoformat(),
                "claim_url": claim_url(token),
                "origin": origin,
            }
        ),
        int(expires_at.timestamp()),
    )


def flat(text: str) -> str:
    """Output with the terminal's line wrapping undone."""
    return " ".join(text.split())


def created_line(token: str = "pat_first") -> list[str]:
    return [
        f"Created a temporary CrewAI account for this project; it expires on {EXPIRY_DATE}.",
        f"Keep it by signing up at {claim_url(token)}, or run `crewai login` to use "
        "your own account instead.",
    ]


# ── crewai deploy ────────────────────────────────────────────────────────────


def test_a_coding_agent_with_no_login_gets_an_account_and_the_deploy_goes_ahead(
    amp, project, coding_agent, capsys
):
    amp.placeholders = [created("pat_first")]
    amp.crews = [201]

    DeployCommand().create_crew(confirm=True, skip_validate=True)

    (asked,) = amp.to(PLACEHOLDERS)
    assert "Authorization" not in asked.headers
    assert "X-Crewai-Organization-Id" not in asked.headers
    assert json.loads(asked.content) == {"source": "deploy", "coding_agent": "claude_code"}
    (deployed,) = amp.to(CREWS)
    assert deployed.headers["Authorization"] == "Bearer pat_first"
    assert deployed.headers["X-Crewai-Organization-Id"] == ORG
    assert saved_token() == "pat_first"
    out = flat(capsys.readouterr().out)
    for line in created_line():
        assert line in out
    assert "Deployment created successfully!" in out
    assert "This used a temporary" not in out


def test_a_person_at_a_terminal_with_no_login_gets_an_account_too(
    amp, project, at_a_terminal, capsys
):
    amp.placeholders = [created("pat_first")]
    amp.crews = [201]

    DeployCommand().create_crew(confirm=True, skip_validate=True)

    (asked,) = amp.to(PLACEHOLDERS)
    assert json.loads(asked.content) == {"source": "deploy", "coding_agent": "unknown"}
    assert amp.bearers(CREWS) == ["Bearer pat_first"]
    assert created_line()[0] in flat(capsys.readouterr().out)


def test_ci_creates_no_account_and_asks_for_a_login(amp, project, capsys):
    with pytest.raises(AuthenticationRequiredError):
        DeployCommand()

    assert amp.requests == []
    assert saved_token() is None
    out = flat(capsys.readouterr().out)
    assert LOGIN_FIRST in out and "Run 'crewai login' to sign up/login." in out
    assert "temporary" not in out


def test_a_real_login_wins_over_any_temporary_account(
    amp, project, coding_agent, capsys
):
    TokenManager().save_tokens("real-login", int(EXPIRES.timestamp()))
    save_placeholder("pat_saved")
    amp.crews = [201]

    DeployCommand().create_crew(confirm=True, skip_validate=True)

    assert amp.to(PLACEHOLDERS) == []
    assert amp.bearers(CREWS) == ["Bearer real-login"]
    assert "temporary" not in capsys.readouterr().out


def test_the_saved_account_is_reused_and_no_second_one_created(
    amp, project, capsys
):
    """Not even CI asks again: the account exists, so it is used."""
    save_placeholder("pat_saved")
    amp.crews = [201]

    DeployCommand().create_crew(confirm=True, skip_validate=True)

    assert amp.to(PLACEHOLDERS) == []
    assert amp.bearers(CREWS) == ["Bearer pat_saved"]
    out = flat(capsys.readouterr().out)
    assert "Created a temporary" not in out
    assert "This used a temporary" not in out


def test_an_expired_account_is_replaced_once(amp, project, coding_agent):
    save_placeholder("pat_old", expires_at=datetime.now(timezone.utc) - timedelta(minutes=1))
    amp.placeholders = [created("pat_new")]
    amp.crews = [201]

    DeployCommand().create_crew(confirm=True, skip_validate=True)

    assert len(amp.to(PLACEHOLDERS)) == 1
    assert amp.bearers(CREWS) == ["Bearer pat_new"]
    assert saved_token() == "pat_new"


def test_an_account_amp_refuses_is_replaced_once_and_the_deploy_retried(
    amp, project, coding_agent
):
    save_placeholder("pat_old")
    amp.placeholders = [created("pat_new")]
    amp.crews = [401, 201]

    result = CliRunner().invoke(deploy_create, ["-y", "--skip-validate"])

    assert result.exit_code == 0, result.output
    assert amp.bearers(CREWS) == ["Bearer pat_old", "Bearer pat_new"]
    assert len(amp.to(PLACEHOLDERS)) == 1
    out = flat(result.output)
    assert (
        "CrewAI AMP no longer accepts the temporary account (it expired or was "
        "removed), so it was discarded."
    ) in out
    assert "This used a temporary" not in out
    assert saved_token() == "pat_new"


def test_a_second_refusal_stops_without_a_loop(amp, project, coding_agent):
    amp.placeholders = [created("pat_first"), created("pat_second")]
    amp.crews = [401, 401]

    result = CliRunner().invoke(deploy_create, ["-y", "--skip-validate"])

    assert result.exit_code == 1
    assert amp.bearers(CREWS) == ["Bearer pat_first", "Bearer pat_second"]
    assert len(amp.to(PLACEHOLDERS)) == 2
    assert saved_token() is None


def test_a_push_amp_refuses_is_retried_with_a_new_account(amp, project, coding_agent):
    save_placeholder("pat_old")
    amp.placeholders = [created("pat_new")]
    amp.crews = [401, 200]

    result = CliRunner().invoke(deploy_push, ["--skip-validate"])

    assert result.exit_code == 0, result.output
    assert amp.bearers(PUSH) == ["Bearer pat_old", "Bearer pat_new"]
    assert "This used a temporary" not in flat(result.output)


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (
            httpx.Response(429, json={"error": "placeholder_limit_reached"}),
            "the limit for new temporary accounts was reached, try again later",
        ),
        (
            httpx.Response(503, json={"error": "placeholder_accounts_disabled"}),
            "temporary accounts are not available right now",
        ),
        (
            httpx.Response(403, json={"error": "placeholder_accounts_blocked"}),
            "temporary accounts are not available from this network",
        ),
        (httpx.Response(404, text="Not Found"), "CrewAI AMP answered 404"),
        (
            created(claim_url="javascript:alert(1)"),
            "CrewAI AMP's answer was not a usable account",
        ),
        (created(token=""), "CrewAI AMP's answer was not a usable account"),
        (
            created(expires_at="2020-01-01T00:00:00Z"),
            "CrewAI AMP's answer was not a usable account",
        ),
    ],
)
def test_amp_not_creating_an_account_falls_back_to_the_login_message(
    amp, project, coding_agent, capsys, answer, reason
):
    amp.placeholders = [answer]

    with pytest.raises(AuthenticationRequiredError):
        DeployCommand()

    assert len(amp.requests) == 1
    assert saved_token() is None
    out = flat(capsys.readouterr().out)
    assert f"A temporary CrewAI account could not be created: {reason}." in out
    assert out.index("could not be created") < out.index(LOGIN_FIRST)
    assert "Run 'crewai login' to sign up/login." in out


def test_an_unreachable_amp_falls_back_to_the_login_message(
    amp, project, coding_agent, capsys
):
    amp.unreachable = True

    with pytest.raises(AuthenticationRequiredError):
        DeployCommand()

    out = flat(capsys.readouterr().out)
    assert (
        "A temporary CrewAI account could not be created: could not reach CrewAI "
        "AMP (ConnectError)."
    ) in out
    assert LOGIN_FIRST in out


def test_the_token_is_never_sent_to_another_amp(
    monkeypatch, amp, project, coding_agent
):
    """Issued by amp.test, so a project pointed at another AMP gets its own."""
    save_placeholder("pat_amp_test")
    monkeypatch.setenv("CREWAI_PLUS_URL", "https://other.test")
    amp.placeholders = [created("pat_other")]
    amp.crews = [201]

    DeployCommand().create_crew(confirm=True, skip_validate=True)

    assert all(
        request.headers.get("Authorization") != "Bearer pat_amp_test"
        for request in amp.requests
    )
    assert {request.url.host for request in amp.requests} == {"other.test"}
    assert amp.bearers(CREWS) == ["Bearer pat_other"]


def test_ci_with_another_amps_account_saved_still_asks_for_a_login(
    monkeypatch, amp, project, capsys
):
    save_placeholder("pat_amp_test")
    monkeypatch.setenv("CREWAI_PLUS_URL", "https://other.test")

    with pytest.raises(AuthenticationRequiredError):
        DeployCommand()

    assert amp.requests == []
    assert LOGIN_FIRST in flat(capsys.readouterr().out)


def test_other_amp_commands_never_create_or_use_an_account(amp, coding_agent, capsys):
    """`crewai org`, `tool`, `triggers` and `skill` share the login check; only
    deploy and `eval --models` were meant to change."""
    save_placeholder("pat_saved")

    with pytest.raises(AuthenticationRequiredError):
        OrganizationCommand()

    assert amp.requests == []
    assert LOGIN_FIRST in flat(capsys.readouterr().out)


# ── the run app's Deploy button ──────────────────────────────────────────────


def test_the_run_apps_deploy_uses_a_new_account_instead_of_a_login(
    monkeypatch, amp, project, coding_agent, capsys
):
    logins: list[bool] = []
    monkeypatch.setattr(
        "crewai_cli.authentication.main.AuthenticationCommand.login",
        lambda self: logins.append(True),
    )
    amp.placeholders = [created("pat_first")]
    amp.crews = [201]

    run_crew._chain_deploy()

    assert logins == []
    assert amp.bearers(CREWS) == ["Bearer pat_first"]
    assert "This used a temporary" not in flat(capsys.readouterr().out)


def test_the_run_apps_deploy_falls_back_to_a_login_when_amp_creates_none(
    monkeypatch, amp, project, coding_agent, capsys
):
    def log_in(self: AuthenticationCommand) -> None:
        TokenManager().save_tokens("real-login", int(EXPIRES.timestamp()))

    monkeypatch.setattr(
        "crewai_cli.authentication.main.AuthenticationCommand.login", log_in
    )
    amp.placeholders = [httpx.Response(503, json={"error": "placeholder_accounts_disabled"})]
    amp.crews = [201]

    run_crew._chain_deploy()

    assert amp.bearers(CREWS) == ["Bearer real-login"]
    out = flat(capsys.readouterr().out)
    assert "temporary accounts are not available right now" in out
    assert "This used a temporary" not in out


# ── crewai login / logout ────────────────────────────────────────────────────


def test_a_successful_login_replaces_the_temporary_account(
    monkeypatch, amp, credentials
):
    save_placeholder("pat_saved")
    monkeypatch.setattr(AuthenticationCommand, "_select_organization", lambda self: None)
    monkeypatch.setattr(
        AuthenticationCommand, "_login_to_tool_repository", lambda self: None
    )
    monkeypatch.setattr(
        AuthenticationCommand,
        "_validate_and_save_token",
        lambda self, data: TokenManager().save_tokens(
            data["access_token"], int(EXPIRES.timestamp())
        ),
    )
    monkeypatch.setattr(
        "crewai_core.auth.oauth2.httpx.post",
        lambda *args, **kwargs: httpx.Response(200, json={"access_token": "real-login"}),
    )
    command = AuthenticationCommand()
    command.oauth2_provider = SimpleNamespace(
        get_token_url=lambda: "https://login.test/token", get_client_id=lambda: "cli"
    )

    command._poll_for_token({"device_code": "code", "interval": 0})

    assert saved_token() is None
    assert TokenManager().get_token() == "real-login"


def test_a_login_that_times_out_keeps_the_temporary_account(
    monkeypatch, amp, credentials
):
    save_placeholder("pat_saved")
    monkeypatch.setattr(
        "crewai_core.auth.oauth2.httpx.post",
        lambda *args, **kwargs: httpx.Response(400, json={"error": "expired_token"}),
    )
    command = AuthenticationCommand()
    command.oauth2_provider = SimpleNamespace(
        get_token_url=lambda: "https://login.test/token", get_client_id=lambda: "cli"
    )

    with pytest.raises(SystemExit):
        command._poll_for_token({"device_code": "code", "interval": 0})

    assert saved_token() == "pat_saved"


def test_logout_forgets_the_temporary_account(credentials):
    save_placeholder("pat_saved")

    result = CliRunner().invoke(logout)

    assert result.exit_code == 0, result.output
    assert saved_token() is None


# ── crewai eval --models ─────────────────────────────────────────────────────

MODELS = "openai/gpt-4o-mini"


@pytest.fixture
def deployed(monkeypatch, project):
    monkeypatch.setattr(eval_module, "get_or_create_project_id", lambda: "proj-1")
    monkeypatch.setattr(eval_module.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(eval_module, "nobody_watching", lambda: False)
    return project


def test_comparing_models_with_no_login_uses_a_new_account(
    amp, deployed, coding_agent, capsys
):
    amp.placeholders = [created("pat_first")]
    amp.evaluations = [202]

    eval_module.eval_models(MODELS, open_report=False)

    (asked,) = amp.to(PLACEHOLDERS)
    assert json.loads(asked.content) == {"source": "eval", "coding_agent": "claude_code"}
    assert amp.bearers(EVALUATIONS) == ["Bearer pat_first"]
    assert amp.bearers(f"{EVALUATIONS}/ev-9") == ["Bearer pat_first"]
    assert amp.to(EVALUATIONS)[0].headers["X-Crewai-Organization-Id"] == ORG
    out = flat(capsys.readouterr().out)
    assert created_line()[0] in out
    assert "This used a temporary" not in out


def test_comparing_models_at_a_terminal_uses_a_new_account(
    amp, deployed, at_a_terminal
):
    amp.placeholders = [created("pat_first")]
    amp.evaluations = [202]

    eval_module.eval_models(MODELS, open_report=False)

    assert amp.bearers(EVALUATIONS) == ["Bearer pat_first"]


def test_comparing_models_in_ci_still_needs_the_login(amp, deployed, capsys):
    with pytest.raises(SystemExit) as stopped:
        eval_module.eval_models(MODELS, open_report=False)

    assert stopped.value.code == 1
    assert amp.requests == []
    out = flat(capsys.readouterr().out)
    assert flat(eval_module.LOGIN_REQUIRED) in out and "temporary" not in out


def test_comparing_models_logged_in_uses_the_login(amp, deployed, coding_agent, capsys):
    TokenManager().save_tokens("real-login", int(EXPIRES.timestamp()))
    save_placeholder("pat_saved")
    amp.evaluations = [202]

    eval_module.eval_models(MODELS, open_report=False)

    assert amp.to(PLACEHOLDERS) == []
    assert amp.bearers(EVALUATIONS) == ["Bearer real-login"]
    assert "temporary" not in capsys.readouterr().out


def test_a_comparison_amp_refuses_the_account_for_is_retried_once(
    amp, deployed, coding_agent, capsys
):
    save_placeholder("pat_old")
    amp.placeholders = [created("pat_new")]
    amp.evaluations = [401, 202]

    eval_module.eval_models(MODELS, open_report=False)

    assert amp.bearers(EVALUATIONS) == ["Bearer pat_old", "Bearer pat_new"]
    assert saved_token() == "pat_new"
    out = flat(capsys.readouterr().out)
    assert "no longer accepts the temporary account" in out
    assert "This used a temporary" not in out


def test_a_second_comparison_refusal_stops_without_a_loop(
    amp, deployed, coding_agent, capsys
):
    amp.placeholders = [created("pat_first"), created("pat_second")]
    amp.evaluations = [401, 401]

    with pytest.raises(SystemExit) as stopped:
        eval_module.eval_models(MODELS, open_report=False)

    assert stopped.value.code == 1
    assert amp.bearers(EVALUATIONS) == ["Bearer pat_first", "Bearer pat_second"]
    assert len(amp.to(PLACEHOLDERS)) == 2


def test_comparing_models_when_amp_creates_no_account_needs_the_login(
    amp, deployed, coding_agent, capsys
):
    amp.placeholders = [httpx.Response(429, json={"error": "placeholder_limit_reached"})]

    with pytest.raises(SystemExit):
        eval_module.eval_models(MODELS, open_report=False)

    out = flat(capsys.readouterr().out)
    assert "A temporary CrewAI account could not be created: the limit" in out
    assert out.index("could not be created") < out.index(flat(eval_module.LOGIN_REQUIRED))
    assert amp.to(EVALUATIONS) == []


def test_comparing_models_on_an_amp_this_machine_does_not_trust_creates_nothing(
    monkeypatch, amp, deployed, coding_agent, capsys
):
    """The project's `.env` chose the AMP: no account is created there, and the
    saved one goes nowhere."""
    save_placeholder("pat_saved")
    monkeypatch.delenv("CREWAI_PLUS_URL")
    (deployed / ".env").write_text("CREWAI_PLUS_URL=https://elsewhere.test\n")
    monkeypatch.setattr(
        eval_module, "Settings", lambda: SimpleNamespace(enterprise_base_url=AMP)
    )

    with pytest.raises(SystemExit):
        eval_module.eval_models(MODELS, open_report=False)

    assert amp.requests == []
    assert flat(eval_module.LOGIN_REQUIRED) in flat(capsys.readouterr().out)
