"""What a login says before the browser opens, and how its end is counted.

The device flow runs for real here; only the provider's HTTP answers, the
browser, the clock, the token store and the telemetry exporter are replaced.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from click.testing import CliRunner

from crewai_cli import run_crew
from crewai_cli.authentication import main as auth_main
from crewai_cli.authentication.main import AuthenticationCommand
from crewai_cli.cli import login as login_command
from crewai_cli.command import AuthenticationRequiredError
from crewai_cli.experimental import eval_crew
from crewai_core import telemetry as core_telemetry
from crewai_core.auth import oauth2

AUTHORIZE_URL = "https://login.example/oauth2/device_authorization"
TOKEN_URL = "https://login.example/oauth2/token"
USER_CODE = "WXYZ-QRST"
DEVICE_CODE = "device-code-secret"
ACCESS_TOKEN = "access-token-secret"
EMAIL = "person@example.com"
DEVICE = {
    "device_code": DEVICE_CODE,
    "user_code": USER_CODE,
    "verification_uri": "https://login.example/activate",
    "verification_uri_complete": f"https://login.example/activate?code={USER_CODE}",
    "interval": 5,
    "expires_in": 900,
}
SOURCES = ["crewai_login", "deploy", "eval", "integrations"]
OPT_OUT_FLAGS = ("CREWAI_DISABLE_TELEMETRY", "OTEL_SDK_DISABLED")
DEPLOY_WHY = (
    "Deploying needs a free CrewAI AMP account. "
    "Your browser will open: sign in or sign up, then come back here."
)
EVAL_WHY = (
    "Evaluating this run needs a free CrewAI AMP account. "
    "Your browser will open: sign in or sign up, then come back here."
)


def started(source: str) -> str:
    return f"cli_usage:login_started:{source}"


def finished(source: str) -> str:
    return f"cli_usage:login_finished:{source}"


def stopped(reason: str, source: str) -> str:
    return f"cli_usage:login_stopped:{reason}:{source}"


def _answer(status: int, body: dict[str, Any], url: str) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("POST", url))


GRANTED = _answer(200, {"access_token": ACCESS_TOKEN, "email": EMAIL}, TOKEN_URL)
PENDING = _answer(400, {"error": "authorization_pending"}, TOKEN_URL)
EXPIRED = _answer(400, {"error": "expired_token"}, TOKEN_URL)
DENIED = _answer(
    400, {"error": "access_denied", "error_description": "User denied"}, TOKEN_URL
)


class _FakeProvider:
    def get_client_id(self) -> str:
        return "client"

    def get_oauth_scopes(self) -> list[str]:
        return ["openid"]

    def get_audience(self) -> str:
        return "audience"

    def get_authorize_url(self) -> str:
        return AUTHORIZE_URL

    def get_token_url(self) -> str:
        return TOKEN_URL

    def get_issuer(self) -> str:
        return "https://login.example"

    def get_jwks_url(self) -> str:
        return "https://login.example/jwks"


class _FakeTokenManager:
    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_tokens(self, access_token: str, expires_at: int) -> None:
        self.saved.append(access_token)


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The provider's answers: the device code (or an error), then one per token
    poll; an exception in either place is raised, as a Ctrl-C or a dropped
    connection would be."""
    answers: dict[str, Any] = {
        "device": _answer(200, DEVICE, AUTHORIZE_URL),
        "polls": [GRANTED],
        "tokens": _FakeTokenManager(),
        "opened": [],
    }

    def post(url: str, **kwargs: Any) -> httpx.Response:
        answer = answers["device"] if url == AUTHORIZE_URL else answers["polls"].pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

    monkeypatch.setattr(oauth2.httpx, "post", post)
    monkeypatch.setattr(oauth2, "time", _Clock())
    monkeypatch.setattr(oauth2.webbrowser, "open", answers["opened"].append)
    monkeypatch.setattr(oauth2, "validate_jwt_token", lambda **kwargs: {"exp": 1})
    monkeypatch.setattr(oauth2, "TokenManager", lambda: answers["tokens"])
    monkeypatch.setattr(
        oauth2.ProviderFactory, "from_settings", classmethod(lambda cls: _FakeProvider())
    )
    # After the token is saved the CLI picks an organization and signs in to the
    # tool repository, both over the network; a test that needs them sets them.
    monkeypatch.setattr(AuthenticationCommand, "_select_organization", lambda self: None)
    monkeypatch.setattr(
        AuthenticationCommand, "_login_to_tool_repository", lambda self: None
    )
    return answers


@pytest.fixture
def usage(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    sent: list[str] = []

    class FakeTelemetry:
        def set_tracer(self) -> None:
            pass

        def feature_usage_span(
            self, feature: str, attributes: dict[str, str] | None = None
        ) -> None:
            assert not attributes, (feature, attributes)
            sent.append(feature)

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", FakeTelemetry)
    return sent


def _said(capsys: pytest.CaptureFixture[str]) -> str:
    return " ".join(capsys.readouterr().out.split())


@pytest.mark.parametrize("source", SOURCES)
def test_a_finished_login_counts_its_start_and_its_finish(
    provider, usage, source
) -> None:
    provider["polls"] = [PENDING, GRANTED]

    AuthenticationCommand().login(source=source)

    assert provider["tokens"].saved == [ACCESS_TOKEN]
    assert usage == [started(source), finished(source)]


def test_a_login_without_a_source_is_counted_as_crewai_login(provider, usage) -> None:
    AuthenticationCommand().login()

    assert usage == [started("crewai_login"), finished("crewai_login")]


@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize(
    ("poll", "reason", "raised"),
    [
        (KeyboardInterrupt(), "abandoned", KeyboardInterrupt),
        (EXPIRED, "timed_out", SystemExit),
        (DENIED, "failed", httpx.HTTPError),
        (httpx.ConnectError("connection reset"), "failed", httpx.ConnectError),
    ],
)
def test_each_way_a_login_stops_is_counted_once_and_still_raised(
    provider, usage, source, poll, reason, raised
) -> None:
    """Counting must not change what the person sees: the error still escapes."""
    provider["polls"] = [PENDING, poll]

    with pytest.raises(raised):
        AuthenticationCommand().login(source=source)

    assert provider["tokens"].saved == []
    assert usage == [started(source), stopped(reason, source)]


def test_a_code_that_runs_out_while_pending_is_a_timeout(provider, usage) -> None:
    provider["device"] = _answer(200, {**DEVICE, "expires_in": 10}, AUTHORIZE_URL)
    provider["polls"] = [PENDING, PENDING, PENDING]

    with pytest.raises(SystemExit) as exc_info:
        AuthenticationCommand().login(source="deploy")

    assert exc_info.value.code == 1
    assert usage == [started("deploy"), stopped("timed_out", "deploy")]


def test_a_device_code_the_provider_refuses_is_a_failed_login(provider, usage) -> None:
    provider["device"] = _answer(503, {"error": "unavailable"}, AUTHORIZE_URL)

    with pytest.raises(httpx.HTTPStatusError, match="503"):
        AuthenticationCommand().login(source="eval")

    assert provider["opened"] == []
    assert usage == [started("eval"), stopped("failed", "eval")]


def test_ctrl_c_at_the_organization_picker_is_still_a_finished_login(
    provider, usage, monkeypatch
) -> None:
    """The token is saved before the picker: quitting it did not lose the login."""

    def quit_the_picker(self: AuthenticationCommand) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(AuthenticationCommand, "_select_organization", quit_the_picker)

    with pytest.raises(KeyboardInterrupt):
        AuthenticationCommand().login(source="deploy")

    assert provider["tokens"].saved == [ACCESS_TOKEN]
    assert usage == [started("deploy"), finished("deploy")]


def test_a_second_login_on_the_same_command_is_counted_on_its_own(
    provider, usage
) -> None:
    command = AuthenticationCommand()
    command.login(source="deploy")
    provider["polls"] = [KeyboardInterrupt()]

    with pytest.raises(KeyboardInterrupt):
        command.login(source="eval")

    assert usage == [
        started("deploy"),
        finished("deploy"),
        started("eval"),
        stopped("abandoned", "eval"),
    ]


@pytest.mark.parametrize(
    ("source", "why"),
    [("deploy", DEPLOY_WHY), ("eval", EVAL_WHY)],
)
def test_why_the_browser_opens_is_said_before_the_link(
    provider, usage, capsys, source, why
) -> None:
    AuthenticationCommand().login(source=source)

    said = _said(capsys)
    assert said.count(why) == 1
    assert said.index(why) < said.index(DEVICE["verification_uri_complete"])
    assert "already filled in" in said
    assert "The code expires in 15 minutes. Press Ctrl-C to cancel." in said


@pytest.mark.parametrize("source", ["crewai_login", "integrations"])
def test_a_login_asked_for_by_name_says_no_more_than_it_did(
    provider, usage, capsys, source
) -> None:
    """`crewai login` is its own reason, and `crewai create` says its own."""
    AuthenticationCommand().login(source=source)

    said = _said(capsys)
    assert "needs a free CrewAI AMP account" not in said
    assert "Your browser will open" not in said
    assert said.startswith("Signing in to CrewAI AMP...")


def test_no_count_carries_a_code_a_token_or_an_email(provider, usage) -> None:
    provider["polls"] = [PENDING, GRANTED]
    AuthenticationCommand().login(source="deploy")
    provider["polls"] = [DENIED]
    with pytest.raises(httpx.HTTPError):
        AuthenticationCommand().login(source="deploy")

    sent = " ".join(usage)
    for secret in (USER_CODE, DEVICE_CODE, ACCESS_TOKEN, EMAIL, "User denied"):
        assert secret not in sent


def test_a_failing_usage_count_never_changes_the_login(
    provider, monkeypatch
) -> None:
    class BrokenTelemetry:
        def set_tracer(self) -> None:
            raise RuntimeError("exporter is down")

    monkeypatch.setattr("crewai_core.telemetry.Telemetry", BrokenTelemetry)

    AuthenticationCommand().login(source="deploy")

    assert provider["tokens"].saved == [ACCESS_TOKEN]


def _replace_the_exporter(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Keep the real `Telemetry` from exporting; returns the span every count
    it sends sets its attributes on."""
    exporter = MagicMock()
    monkeypatch.setattr(core_telemetry, "TracerProvider", lambda **kwargs: exporter)
    monkeypatch.setattr(core_telemetry, "SafeOTLPSpanExporter", MagicMock())
    monkeypatch.setattr(core_telemetry, "BatchSpanProcessor", MagicMock())
    monkeypatch.setattr(core_telemetry, "common_span_attributes", dict)
    monkeypatch.setattr(
        core_telemetry.Telemetry, "_register_shutdown_handlers", lambda self: None
    )
    monkeypatch.setattr(core_telemetry.Telemetry, "_instance", None)
    return exporter.get_tracer.return_value.start_span.return_value


@pytest.mark.parametrize("flag", OPT_OUT_FLAGS)
def test_an_opted_out_process_sends_no_login_count(
    provider, monkeypatch, flag
) -> None:
    """The real `Telemetry`, its exporter replaced: opted out, no span starts;
    opted in, the same login sends its start and its finish."""
    span = _replace_the_exporter(monkeypatch)

    def sent() -> list[str]:
        return [
            call.args[1]
            for call in span.set_attribute.call_args_list
            if call.args[0] == "feature"
        ]

    for name in OPT_OUT_FLAGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(flag, "true")
    AuthenticationCommand().login(source="deploy")
    assert sent() == []

    monkeypatch.delenv(flag)
    monkeypatch.setattr(core_telemetry.Telemetry, "_instance", None)
    provider["polls"] = [GRANTED]
    AuthenticationCommand().login(source="deploy")
    assert sent() == [started("deploy"), finished("deploy")]


class _Deploy:
    """The run app's Deploy: logged out at first, then whatever `creates` says."""

    creates: list[BaseException | None] = []

    def create_crew(self, **kwargs: Any) -> None:
        outcome = _Deploy.creates.pop(0)
        if outcome is not None:
            raise outcome


@pytest.mark.parametrize(
    ("poll", "raised", "deploy_reason", "login_end"),
    [
        (KeyboardInterrupt(), KeyboardInterrupt, "login_abandoned", "abandoned"),
        (EXPIRED, SystemExit, "login_timed_out", "timed_out"),
        (DENIED, httpx.HTTPError, "login_failed", "failed"),
    ],
)
def test_the_run_apps_deploy_keeps_its_own_counts_beside_the_logins(
    provider, usage, monkeypatch, poll, raised, deploy_reason, login_end
) -> None:
    """Dashboards read `deploy_login_prompted` and `deploy_stopped:*`: unchanged."""
    monkeypatch.setattr(_Deploy, "creates", [AuthenticationRequiredError(1)])
    monkeypatch.setattr("crewai_cli.deploy.main.DeployCommand", _Deploy)
    provider["polls"] = [poll]

    with pytest.raises(raised):
        run_crew._chain_deploy()

    assert usage == [
        "cli_usage:deploy_login_prompted",
        started("deploy"),
        stopped(login_end, "deploy"),
        f"cli_usage:deploy_stopped:{deploy_reason}",
    ]


def test_a_deploy_login_that_finishes_then_deploys(
    provider, usage, monkeypatch, capsys
) -> None:
    monkeypatch.setattr(_Deploy, "creates", [AuthenticationRequiredError(1), None])
    monkeypatch.setattr("crewai_cli.deploy.main.DeployCommand", _Deploy)

    run_crew._chain_deploy()

    assert usage == [
        "cli_usage:deploy_login_prompted",
        started("deploy"),
        finished("deploy"),
    ]
    assert DEPLOY_WHY in _said(capsys)


def test_an_evaluations_login_is_counted_as_eval(provider, usage, capsys) -> None:
    provider["polls"] = [KeyboardInterrupt()]

    with pytest.raises(eval_crew.EvaluationStoppedError) as exc_info:
        eval_crew.log_in_again()

    assert exc_info.value.reason == "login_abandoned"
    assert usage == [
        "cli_usage:eval_login_prompted",
        started("eval"),
        stopped("abandoned", "eval"),
    ]
    said = _said(capsys)
    assert said.index(eval_crew.LOGGING_IN_AGAIN) < said.index(EVAL_WHY)


def test_crewai_login_is_counted_as_crewai_login(
    provider, usage, monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("crewai_cli.cli.Settings", MagicMock())

    result = CliRunner().invoke(login_command)

    assert result.exit_code == 0, result.output
    assert usage == [started("crewai_login"), finished("crewai_login")]
    assert "needs a free CrewAI AMP account" not in result.output


def test_the_source_words_are_the_ones_counted() -> None:
    """The warehouse keeps the feature string only, so the words are the API."""
    assert list(auth_main.LoginSource.__args__) == SOURCES
    assert set(auth_main.WHY_LOG_IN) == {"deploy", "eval"}
