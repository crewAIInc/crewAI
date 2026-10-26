"""`crewai doctor`: the project's CrewAI AMP state, one line per check.

Every check is read-only. The only network call is the GET a crew makes for
its `apps=[...]` actions, and only for selectors named with `--app`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
import warnings

import click
from crewai_core.auth.token import AuthError, get_auth_token
from crewai_core.plus_api import PlusAPI
from dotenv import dotenv_values

from crewai_cli.config import Settings
from crewai_cli.run_crew import load_project_env
from crewai_cli.user_data import has_user_declined_tracing, is_tracing_enabled


Status = Literal["ok", "warn", "fail", "skip"]
Line = tuple[Status, str, str]

TRACING_ENV_VAR = "CREWAI_TRACING_ENABLED"
INTEGRATION_TOKEN_ENV = "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN"  # noqa: S105 - a variable name, not a secret


def _check_login(settings: Settings) -> tuple[Line, bool]:
    """The login line, and whether a login could be read."""
    try:
        get_auth_token()
    except AuthError:
        return (
            "fail",
            "login",
            "not logged in (no saved login, or it expired); run `crewai login`",
        ), False
    except Exception as error:
        # Fail closed: a login that exists but cannot be read (a rotated key, a
        # half-written store) is a failed check. Only the type is printed.
        return (
            "fail",
            "login",
            f"the saved login could not be read ({type(error).__name__}); "
            "run `crewai login` again",
        ), False
    if not settings.org_uuid:
        return (
            "fail",
            "login",
            "logged in, but no organization is selected; run `crewai org list`, "
            "then `crewai org switch <id>`",
        ), True
    return (
        "ok",
        "login",
        f"logged in, organization {settings.org_name} ({settings.org_uuid})",
    ), True


def _check_tracing(env_source: str, logged_in: bool, settings: Settings) -> Line:
    env_value = os.environ.get(TRACING_ENV_VAR, "")
    enabled = is_tracing_enabled()
    if env_value.lower() in ("true", "1", "false", "0"):
        reason = f"{TRACING_ENV_VAR}={env_value} in {env_source}"
    elif enabled:
        reason = "consent recorded on this machine"
    elif has_user_declined_tracing():
        reason = "declined on this machine"
    else:
        reason = "not set; the first run at a terminal asks"

    if not enabled:
        # Never a failure: off is the person's choice, not the agent's to flip.
        return (
            "warn",
            "tracing",
            f"off ({reason}); runs send no traces. "
            "Ask the user before running `crewai traces enable`",
        )
    host = PlusAPI().base_url.rstrip("/")
    if logged_in and settings.org_uuid:
        destination = f"{host}, organization {settings.org_name}"
    elif logged_in:
        destination = f"{host}, no organization selected"
    else:
        destination = f"{host} without a login (anonymous)"
    return "ok", "tracing", f"on ({reason}); traces go to {destination}"


def _check_integrations(apps: tuple[str, ...]) -> list[Line]:
    if not apps:
        return [
            (
                "skip",
                "integration",
                "no --app given; check a selector with `crewai doctor --app gmail`",
            )
        ]
    if not os.environ.get(INTEGRATION_TOKEN_ENV):
        return [
            (
                "fail",
                f"integration {app}",
                f"{INTEGRATION_TOKEN_ENV} is not set in the environment or .env",
            )
            for app in apps
        ]
    try:
        # Importing crewai_tools initializes optional tool SDKs; their
        # import-time warnings say nothing about this check.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from crewai_tools.tools.crewai_platform_tools.integrations_client import (
                ApplicationSelector,
                client_for_selector,
            )
        import requests
    except ImportError:
        return [
            (
                "fail",
                f"integration {app}",
                "crewai-tools is not installed for this CLI; run `uv run crewai "
                "doctor` in a project that depends on crewai[tools]",
            )
            for app in apps
        ]

    lines: list[Line] = []
    for app in apps:
        check = f"integration {app}"
        try:
            selector = ApplicationSelector.from_string(app)
        except ValueError as error:
            lines.append(("fail", check, str(error)))
            continue
        # The same two calls `CrewaiPlatformTools` makes for `apps=[...]`.
        try:
            actions = client_for_selector(selector).get_actions([selector])
        except requests.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status in (401, 403):
                detail = (
                    f"{INTEGRATION_TOKEN_ENV} is invalid or expired (HTTP {status})"
                )
            else:
                detail = f"AMP answered HTTP {status}"
            lines.append(("fail", check, detail))
            continue
        except requests.RequestException as error:
            lines.append(
                ("fail", check, f"could not reach AMP ({type(error).__name__})")
            )
            continue
        except Exception as error:
            # Fail closed on anything unclassified, such as a response shape
            # this CLI does not know.
            lines.append(("fail", check, f"{type(error).__name__}: {error}"))
            continue
        if not actions:
            # AMP answers an empty list for an app it does not know, one that is
            # not connected, and its own errors alike: the three cannot be told apart.
            lines.append(
                (
                    "fail",
                    check,
                    "no actions matched: not connected, misspelled, "
                    "or AMP could not list it",
                )
            )
        else:
            slugs = ", ".join(action.action for action in actions)
            lines.append(("ok", check, f"{len(actions)} actions: {slugs}"))
    return lines


def doctor(apps: tuple[str, ...]) -> None:
    """Print one `<status> <check>: <detail>` line per check; exit 1 on a fail."""
    env_file = Path.cwd() / ".env"
    env_source = "the environment"
    if env_file.is_file():
        if TRACING_ENV_VAR in dotenv_values(env_file):
            env_source = ".env"
        load_project_env(env_file)

    settings = Settings()
    login_line, logged_in = _check_login(settings)
    lines = [
        login_line,
        _check_tracing(env_source, logged_in, settings),
        *_check_integrations(apps),
    ]
    for status, check, detail in lines:
        click.echo(f"{status} {check}: {detail}")
    if any(status == "fail" for status, _, _ in lines):
        raise SystemExit(1)
