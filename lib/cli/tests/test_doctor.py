"""`crewai doctor`: one line per check, exit 1 when any check fails, and
nothing sent but the GET a crew makes for its `apps=[...]` actions."""

from __future__ import annotations

from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import sys

from click.testing import CliRunner
from crewai_cli.cli import doctor
from crewai_core.token_manager import TokenManager
from crewai_core.user_data import update_user_data
from crewai_tools.tools.crewai_platform_tools import integrations_client
from crewai_tools.tools.crewai_platform_tools.integrations_client import ToolInfo
import httpx
import pytest
import requests


TOKEN_ENV = "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN"
PAT_ENV = "CREWAI_USER_PAT"
TOKEN_MODULE = "crewai_tools.tools.crewai_platform_tools.integrations_client"


@pytest.fixture(autouse=True)
def machine(tmp_path, monkeypatch):
    """A machine of the test's own: saved login, CLI settings and user data
    under tmp_path, run from an empty project directory. `crewai doctor` loads
    the project's `.env` into the process, so the environment is put back."""
    saved_environment = dict(os.environ)
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    for name in ("CREWAI_TRACING_ENABLED", TOKEN_ENV, "CREWAI_PLUS_URL", PAT_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CREWAI_STORAGE_DIR", "crewai_doctor_test")
    monkeypatch.setattr(
        "crewai_core.paths.appdirs.user_data_dir",
        lambda app, author: str(tmp_path / "data" / app),
    )
    credentials = tmp_path / "credentials"
    credentials.mkdir()
    monkeypatch.setattr(
        TokenManager, "_get_secure_storage_path", staticmethod(lambda: credentials)
    )
    settings_file = tmp_path / "settings.json"
    monkeypatch.setattr(
        "crewai_core.settings.get_writable_config_path", lambda: settings_file
    )
    yield project
    os.environ.clear()
    os.environ.update(saved_environment)


def log_in(*, expires_in: timedelta = timedelta(hours=1)) -> None:
    TokenManager().save_tokens(
        "login-token", int((datetime.now() + expires_in).timestamp())
    )


def select_org(tmp_path: Path) -> None:
    (tmp_path / "settings.json").write_text(
        json.dumps({"org_name": "Acme", "org_uuid": "org-uuid-1"})
    )


def logged_in_with_org(tmp_path: Path) -> None:
    log_in()
    select_org(tmp_path)


def lines_of(output: str) -> dict[str, str]:
    """`<status> <check>: <detail>` lines, keyed by check."""
    lines: dict[str, str] = {}
    for line in output.splitlines():
        status, _, rest = line.partition(" ")
        check, _, detail = rest.partition(": ")
        lines[check] = f"{status} {detail}"
    return lines


def run(*args: str):
    return CliRunner().invoke(doctor, list(args))


class FakeClient:
    def __init__(self, result):
        self.result = result
        self.selectors: list = []

    def get_actions(self, selectors):
        self.selectors.extend(selectors)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def answer_selectors(monkeypatch, results: dict):
    """`client_for_selector` answers each app with its result; the token the
    call saw is recorded."""
    seen_tokens: list[str | None] = []

    def client_for_selector(selector, *, base_url=None):
        seen_tokens.append(os.environ.get(TOKEN_ENV))
        return FakeClient(results[selector.app])

    monkeypatch.setattr(integrations_client, "client_for_selector", client_for_selector)
    return seen_tokens


def actions(app: str, *slugs: str) -> list[ToolInfo]:
    return [
        ToolInfo(
            app=app, action=slug, connection_id=None, description="", parameters={}
        )
        for slug in slugs
    ]


def http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    return requests.HTTPError(f"{status} Client Error", response=response)


# login


def test_logged_out_fails_with_a_login_hint():
    result = run()

    assert result.exit_code == 1
    assert lines_of(result.output)["login"] == (
        "fail not logged in (no saved login, or it expired); run `crewai login`"
    )


def test_logged_in_with_an_organization_is_ok(tmp_path):
    logged_in_with_org(tmp_path)

    result = run()

    assert result.exit_code == 0, result.output
    assert lines_of(result.output)["login"] == (
        "ok logged in, organization Acme (org-uuid-1)"
    )


def test_logged_in_without_an_organization_fails_with_a_switch_hint():
    log_in()

    result = run()

    assert result.exit_code == 1
    assert lines_of(result.output)["login"].startswith(
        "fail logged in, but no organization is selected"
    )
    assert "crewai org switch <id>" in result.output


def test_an_expired_login_fails_as_not_logged_in(tmp_path):
    log_in(expires_in=timedelta(hours=-1))
    select_org(tmp_path)

    result = run()

    assert result.exit_code == 1
    assert lines_of(result.output)["login"].startswith("fail not logged in")


def test_an_unreadable_login_fails_naming_the_error_type(tmp_path):
    log_in()
    select_org(tmp_path)
    (tmp_path / "credentials" / "tokens.enc").write_bytes(b"not a fernet token")

    result = run()

    assert result.exit_code == 1
    assert lines_of(result.output)["login"] == (
        "fail the saved login could not be read (InvalidToken); "
        "run `crewai login` again"
    )


# tracing


def test_tracing_turned_on_in_the_project_env_is_seen(tmp_path, machine):
    (machine / ".env").write_text("CREWAI_TRACING_ENABLED=true\n")

    result = run()

    assert lines_of(result.output)["tracing"] == (
        "ok on (CREWAI_TRACING_ENABLED=true in .env); "
        "traces go to https://app.crewai.com without a login (anonymous)"
    )


def test_tracing_on_while_logged_in_names_the_organization(tmp_path, monkeypatch):
    logged_in_with_org(tmp_path)
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "1")

    result = run()

    assert lines_of(result.output)["tracing"] == (
        "ok on (CREWAI_TRACING_ENABLED=1 in the environment); "
        "traces go to https://app.crewai.com, organization Acme"
    )


def test_tracing_off_in_the_project_env_warns_and_does_not_fail(tmp_path, machine):
    logged_in_with_org(tmp_path)
    update_user_data({"trace_consent": True, "first_execution_done": True})
    (machine / ".env").write_text("CREWAI_TRACING_ENABLED=false\n")

    result = run()

    assert result.exit_code == 0, result.output
    assert lines_of(result.output)["tracing"] == (
        "warn off (CREWAI_TRACING_ENABLED=false in .env); runs send no traces. "
        "Ask the user before setting CREWAI_TRACING_ENABLED=true in .env"
    )


def test_tracing_off_in_the_shell_points_at_the_environment(tmp_path, monkeypatch):
    """`crewai traces enable` only records consent, which a false here beats."""
    update_user_data({"trace_consent": True, "first_execution_done": True})
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "0")

    result = run()

    assert lines_of(result.output)["tracing"] == (
        "warn off (CREWAI_TRACING_ENABLED=0 in the environment); runs send no "
        "traces. Ask the user before setting CREWAI_TRACING_ENABLED=true in the "
        "environment"
    )


def test_a_valueless_key_in_the_project_env_leaves_the_shell_value(
    tmp_path, machine, monkeypatch
):
    (machine / ".env").write_text("CREWAI_TRACING_ENABLED\n")
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")

    result = run()

    assert lines_of(result.output)["tracing"].startswith(
        "ok on (CREWAI_TRACING_ENABLED=true in the environment)"
    )


def test_tracing_names_the_integration_token_a_run_would_send(tmp_path, machine):
    """Runs send CREWAI_USER_PAT, then the integration token, then the login."""
    (machine / ".env").write_text(
        f"CREWAI_TRACING_ENABLED=true\n{TOKEN_ENV}=integration-token\n"
    )

    result = run()

    assert lines_of(result.output)["tracing"] == (
        "ok on (CREWAI_TRACING_ENABLED=true in .env); "
        f"traces go to https://app.crewai.com with {TOKEN_ENV}"
    )
    assert "integration-token" not in result.output


def test_tracing_names_the_pat_over_the_login(tmp_path, monkeypatch):
    logged_in_with_org(tmp_path)
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    monkeypatch.setenv(PAT_ENV, "pat-token")
    monkeypatch.setenv(TOKEN_ENV, "integration-token")

    result = run()

    assert lines_of(result.output)["tracing"] == (
        "ok on (CREWAI_TRACING_ENABLED=true in the environment); "
        f"traces go to https://app.crewai.com with {PAT_ENV}"
    )
    assert "pat-token" not in result.output
    assert "login-token" not in result.output


def test_tracing_consent_recorded_is_on(tmp_path):
    update_user_data({"trace_consent": True, "first_execution_done": True})

    result = run()

    assert lines_of(result.output)["tracing"].startswith(
        "ok on (consent recorded on this machine)"
    )


def test_tracing_declined_warns(tmp_path):
    update_user_data({"trace_consent": False, "first_execution_done": True})

    result = run()

    assert lines_of(result.output)["tracing"] == (
        "warn off (declined on this machine); runs send no traces. "
        "Ask the user before running `crewai traces enable`"
    )


def test_tracing_never_set_says_the_first_run_asks():
    result = run()

    assert lines_of(result.output)["tracing"].startswith(
        "warn off (not set; the first run at a terminal asks)"
    )


# integrations


def test_without_app_the_integration_check_is_skipped(tmp_path):
    logged_in_with_org(tmp_path)

    result = run()

    assert result.exit_code == 0
    assert lines_of(result.output)["integration"] == (
        "skip no --app given; check a selector with `crewai doctor --app gmail`"
    )


def test_an_app_without_the_integration_token_fails(tmp_path, monkeypatch):
    logged_in_with_org(tmp_path)
    answer_selectors(monkeypatch, {})

    result = run("--app", "gmail")

    assert result.exit_code == 1
    assert lines_of(result.output)["integration gmail"] == (
        f"fail {TOKEN_ENV} is not set in the environment or .env"
    )


def test_the_integration_token_is_read_from_the_project_env(
    tmp_path, machine, monkeypatch
):
    logged_in_with_org(tmp_path)
    (machine / ".env").write_text(f"{TOKEN_ENV}=from-dotenv\n")
    seen_tokens = answer_selectors(
        monkeypatch, {"gmail": actions("gmail", "gmail_send_email")}
    )

    result = run("--app", "gmail")

    assert result.exit_code == 0, result.output
    assert seen_tokens == ["from-dotenv"]


def test_an_app_with_actions_lists_their_slugs(tmp_path, monkeypatch):
    logged_in_with_org(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "integration-token")
    answer_selectors(
        monkeypatch,
        {"gmail": actions("gmail", "gmail_send_email", "gmail_get_thread")},
    )

    result = run("--app", "gmail")

    assert result.exit_code == 0, result.output
    assert lines_of(result.output)["integration gmail"] == (
        "ok 2 actions: gmail_send_email, gmail_get_thread"
    )


@pytest.mark.parametrize(
    "answer,detail",
    [
        (
            [],
            "no actions matched: not connected, misspelled, or AMP could not list it",
        ),
        (http_error(401), f"{TOKEN_ENV} is invalid or expired (HTTP 401)"),
        (http_error(403), f"{TOKEN_ENV} is invalid or expired (HTTP 403)"),
        (http_error(500), "AMP answered HTTP 500"),
        (requests.ConnectionError("refused"), "could not reach AMP (ConnectionError)"),
        (
            requests.JSONDecodeError("Expecting value", "<html>login</html>", 0),
            "AMP answered with something that is not JSON; check CREWAI_PLUS_URL",
        ),
        (KeyError("data"), "KeyError: 'data'"),
    ],
)
def test_an_app_that_cannot_be_listed_fails(tmp_path, monkeypatch, answer, detail):
    logged_in_with_org(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "integration-token")
    answer_selectors(monkeypatch, {"gmail": answer})

    result = run("--app", "gmail")

    assert result.exit_code == 1
    assert lines_of(result.output)["integration gmail"] == f"fail {detail}"


def test_a_malformed_selector_fails_with_the_parse_error(tmp_path, monkeypatch):
    logged_in_with_org(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "integration-token")
    answer_selectors(monkeypatch, {})

    result = run("--app", "gmail@")

    assert result.exit_code == 1
    assert lines_of(result.output)["integration gmail@"] == (
        "fail Invalid application selector 'gmail@': connection ID cannot be empty"
    )


def test_each_app_gets_its_own_line(tmp_path, monkeypatch):
    logged_in_with_org(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "integration-token")
    answer_selectors(
        monkeypatch,
        {"gmail": actions("gmail", "gmail_send_email"), "slack": []},
    )

    result = run("--app", "gmail", "--app", "slack")

    assert result.exit_code == 1
    lines = lines_of(result.output)
    assert lines["integration gmail"] == "ok 1 action: gmail_send_email"
    assert lines["integration slack"].startswith("fail no actions matched")


def test_without_crewai_tools_an_app_fails_with_the_uv_run_hint(
    tmp_path, monkeypatch
):
    logged_in_with_org(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "integration-token")
    monkeypatch.setitem(sys.modules, TOKEN_MODULE, None)

    result = run("--app", "gmail")

    assert result.exit_code == 1
    assert lines_of(result.output)["integration gmail"] == (
        "fail crewai-tools is not installed for this CLI; run `uv run crewai "
        "doctor` in a project that depends on crewai[tools]"
    )


# what doctor sends and writes


def test_only_the_actions_get_is_sent_with_the_integration_token(
    tmp_path, machine, monkeypatch
):
    """The real client runs; every request other than the actions GET would
    raise. The login token never leaves the machine."""
    logged_in_with_org(tmp_path)
    monkeypatch.setenv(TOKEN_ENV, "integration-token")
    gets: list[tuple[str, dict, dict]] = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self):
            return {"actions": {"gmail": [{"name": "gmail_send_email"}]}}

    def get(url, *, headers, params, **kwargs):
        gets.append((url, headers, params))
        return Response()

    def refuse(*args, **kwargs):
        raise AssertionError("crewai doctor sent a request other than the GET")

    monkeypatch.setattr(requests, "get", get)
    for name in ("post", "put", "patch", "delete"):
        monkeypatch.setattr(requests, name, refuse)
    monkeypatch.setattr(requests.Session, "request", refuse)
    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse)

    result = run("--app", "gmail")

    assert result.exit_code == 0, result.output
    assert lines_of(result.output)["integration gmail"] == (
        "ok 1 action: gmail_send_email"
    )
    assert gets == [
        (
            "https://app.crewai.com/crewai_plus/api/v1/integrations/actions",
            {"Authorization": "Bearer integration-token"},
            {"apps": "gmail"},
        )
    ]


def test_the_project_files_are_left_as_they_were(tmp_path, machine):
    pyproject = machine / "pyproject.toml"
    pyproject.write_text('[project]\nname = "demo"\n\n[tool.crewai]\ntype = "crew"\n')
    before = pyproject.read_bytes()

    run()

    assert pyproject.read_bytes() == before
    assert sorted(path.name for path in machine.iterdir()) == ["pyproject.toml"]
