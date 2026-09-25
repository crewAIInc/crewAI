"""`crewai eval`: evaluate the last traced run through CrewAI AMP.

crewAI records a traced run in `.crewai/last_run.json` when the run's spans
reach Wharf. This command reads that record (or takes `--run EXECUTION_ID`),
asks AMP to evaluate the run, prints and opens the URL AMP answers with,
waits for the verdict and prints it. With no traced run recorded it offers
to turn tracing on for the project and run the crew now.

Who may evaluate what is AMP's decision: an anonymous run once without an
account, then it needs one; a run traced while logged in for that
organization's members; a deployment execution for members who may see its
traces. The command sends the saved `crewai login` when there is one.
"""

from __future__ import annotations

import contextlib
from ipaddress import ip_address
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlparse
import webbrowser

import click
from crewai_core.constants import DEFAULT_CREWAI_ENTERPRISE_URL
from crewai_core.settings import Settings
from dotenv import load_dotenv, set_key
import httpx
from rich.console import Console
from rich.text import Text

from crewai_cli.authentication.token import AuthError, get_auth_token
from crewai_cli.plus_api import PlusAPI
from crewai_cli.utils import get_or_create_project_id, is_dmn_mode_enabled


console = Console()

LAST_RUN_FILE = Path(".crewai") / "last_run.json"
TRACING_ENV_VAR = "CREWAI_TRACING_ENABLED"
POLL_SECONDS = 3.0
POLL_RETRIES = 5  # consecutive unreachable / 5xx polls before giving up; the evaluation keeps running
FINISHED = {"done", "failed"}
STATUSES = {"queued", "running"} | FINISHED


def _record_usage(execution_id: str, *, logged_in: bool) -> None:
    """Count an evaluation that is actually starting, and what can be joined on.

    The project id, the runtime and the version ride every span already. What
    only this command knows is WHICH run is being graded, whether the caller was
    logged in, and — when they are — which organization they are logged in to.
    Nothing about the run's content is recorded here: no inputs, no output, no
    verdict. Those live in AMP, which the execution id joins to.

    The TUI's button counts `cli_usage:evaluate` when it is pressed, so the
    difference between that and `cli_usage:eval` is intent that never became an
    evaluation.
    """
    try:
        from crewai_core.settings import Settings
        from crewai_core.telemetry import Telemetry

        organization = (
            str(getattr(Settings(), "org_uuid", "") or "") if logged_in else ""
        )
        telemetry = Telemetry()
        telemetry.set_tracer()
        telemetry.feature_usage_span(
            "cli_usage:eval",
            {
                "execution_id": execution_id,
                "authenticated": "true" if logged_in else "false",
                "organization_id": organization,
            },
        )
    except Exception:  # noqa: S110 - telemetry must never break a command
        pass


def eval_crew(run_id: str | None = None) -> None:
    """Evaluate the last traced run of this project, or the run RUN_ID."""
    get_or_create_project_id()
    # Read before the project's .env is loaded, so a project cannot add itself.
    trusted = _trusted_amp_origins()
    _load_project_env()
    record = read_last_run() or {}
    execution_id = run_id or record.get("execution_id")
    if execution_id is None:
        execution_id = _run_now_or_explain()
        record = read_last_run() or {}

    client = _amp_client(trusted)
    recorded_amp = str(record.get("amp_base_url") or "").rstrip("/")
    if not run_id and recorded_amp and recorded_amp != client.base_url.rstrip("/"):
        console.print(
            Text(
                f"The run was traced to {recorded_amp}; evaluating at the configured AMP {client.base_url}."
            ),
            style="yellow",
        )
    _record_usage(execution_id, logged_in=client.api_key is not None)
    started = _start_evaluation(client, execution_id)
    url = started.get("url")
    console.print(Text("Evaluating run ").append(execution_id, style="bold"))
    if url:
        # Appended, never interpolated: this line invites a click, so a `url`
        # carrying `[link=…]` would print a trustworthy label over a hostile
        # target. The style belongs to the span, not to the string.
        console.print(Text("Follow it at ").append(url, style="cyan underline"))
        _open(url)

    finished = _wait(client, started["id"], url)
    _print_verdict(finished, url)
    if finished.get("status") != "done":
        raise SystemExit(1)


def _trusted_amp_origins() -> set[str]:
    """Where the saved login may be sent: the AMP this machine is configured for
    (`crewai enterprise configure`), one already exported in this shell, and
    crewAI's own."""
    candidates = (
        os.environ.get("CREWAI_PLUS_URL"),
        Settings().enterprise_base_url,
        DEFAULT_CREWAI_ENTERPRISE_URL,
    )
    return {origin for origin in map(_origin, candidates) if origin}


def _origin(url: str | None) -> str | None:
    parsed = urlparse(str(url or ""))
    return (
        f"{parsed.scheme}://{parsed.netloc}".lower()
        if parsed.scheme and parsed.netloc
        else None
    )


def _encrypted(origin: str | None) -> bool:
    """HTTPS, or plain HTTP to this machine — the rule `TraceGrantClient` already
    applies to collector grants (localhost, its subdomains, loopback addresses)."""
    parsed = urlparse(origin or "")
    if parsed.scheme == "https":
        return True
    if parsed.scheme != "http":
        return False
    hostname = (parsed.hostname or "").rstrip(".")
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def _amp_client(trusted: set[str]) -> PlusAPI:
    """The AMP to ask, and whether the saved login goes with it.

    A project's `.env` may point `crewai eval` at another AMP — that is how a
    self-hosted project is wired, and the run was traced there — so the request
    follows it. The credential does not: it goes only to an AMP this machine is
    logged in to, over a connection that encrypts it. Anywhere else the run is
    read anonymously."""
    client = PlusAPI(api_key=saved_login())
    if client.api_key is None:
        return client

    origin = _origin(client.base_url)
    if origin in trusted and _encrypted(origin):
        return client

    why = (
        "is not an AMP this machine is logged in to"
        if origin not in trusted
        else "would carry the login over plain HTTP"
    )
    console.print(
        Text(
            f"Reading anonymously: {client.base_url} {why}. "
            "Run `crewai enterprise configure <url>` to log in to it."
        ),
        style="yellow",
    )
    return PlusAPI()


def _load_project_env() -> None:
    """The project's .env, as `crewai run` loads it — so CREWAI_PLUS_URL here is the one the run used."""
    env_file = Path.cwd() / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=True)


def read_last_run(directory: Path | None = None) -> dict[str, Any] | None:
    """The record crewAI wrote for the project's last traced run (execution_id,
    tier, started_at, finished_at, amp_base_url), or None."""
    path = (directory or Path.cwd()) / LAST_RUN_FILE
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(loaded, dict) or not loaded.get("execution_id"):
        return None
    loaded["execution_id"] = str(loaded["execution_id"])
    return loaded


def saved_login() -> str | None:
    """The `crewai login` token, or None: AMP then treats the caller as anonymous.

    Only "not logged in" reads as anonymous. A credential that exists but cannot
    be read — a rotated key, a half-written store, a directory `sudo` left owned
    by root — is said out loud: reading anonymously instead would quietly spend
    the run's one anonymous read and then refuse a user who believes they are
    logged in."""
    try:
        return get_auth_token()
    except AuthError:
        return None
    except Exception as error:
        _fail(
            f"Could not read the saved login ({type(error).__name__}: {error}). "
            "Run `crewai login` again, or `crewai eval` will not know who you are."
        )
        return None


def _run_now_or_explain() -> str:
    """No traced run recorded here: offer to turn tracing on and run the crew now."""
    if not Path("pyproject.toml").is_file():
        _fail(
            "No crewAI project here (no pyproject.toml). Run `crewai eval` from the project's "
            "directory, or name a run: `crewai eval --run EXECUTION_ID`."
        )
    steps = (
        "No traced run is recorded in this project. Turn tracing on — its runs are then "
        "traced to CrewAI AMP — and run the crew, then come back:\n"
        f"  1. add {TRACING_ENV_VAR}=true to .env\n  2. crewai run\n  3. crewai eval"
    )
    if is_dmn_mode_enabled() or not sys.stdin.isatty():
        console.print(steps, style="yellow")
        raise SystemExit(1)
    if not click.confirm(
        "No traced run is recorded in this project. Turn tracing on and run the crew now? "
        "The run's trace is sent to CrewAI AMP.",
        default=True,  # João, 2026-09-20: y/n with Y as the default — the prompt says what Enter does
    ):
        console.print(steps, style="yellow")
        raise SystemExit(0)

    _enable_tracing()
    from crewai_cli.run_crew import run_crew

    run_crew()
    record = read_last_run()
    if record is None:
        console.print(
            "The run finished but no trace was recorded: the run may have failed, sharing the "
            "trace was declined, or this project's crewai is older than the version that records "
            f"the last run ({LAST_RUN_FILE}). Run the crew again and accept when asked, then `crewai eval`.",
            style="bold red",
        )
        raise SystemExit(1)
    return str(record["execution_id"])


def _enable_tracing() -> None:
    """`CREWAI_TRACING_ENABLED=true` in the project's .env, and in this process for the run about to start."""
    env_file = Path.cwd() / ".env"
    env_file.touch(exist_ok=True)
    set_key(str(env_file), TRACING_ENV_VAR, "true", quote_mode="never")
    os.environ[TRACING_ENV_VAR] = "true"
    console.print(
        "Tracing is on for this project — its runs are traced to CrewAI AMP.",
        style="green",
    )


def _start_evaluation(client: PlusAPI, execution_id: str) -> dict[str, Any]:
    try:
        response = client.create_evaluation(execution_id)
    except httpx.HTTPError as error:
        _fail(f"Could not reach AMP to start the evaluation: {error}")
    if response.status_code in (200, 202):
        payload = _payload(response)
        # The id is what every later call is made with, so a missing or
        # non-string one is a protocol error and not something to carry on
        # with. The url is only ever shown and opened, so a malformed one
        # costs the link and nothing else: the evaluation is already running
        # and its verdict is what the user came for.
        if payload and isinstance(payload.get("id"), str) and payload["id"]:
            if not isinstance(payload.get("url"), str):
                if payload.get("url") is not None:
                    console.print(
                        Text(
                            "AMP answered with a report url that is not a string; "
                            "the link is unavailable for this run."
                        ),
                        style="yellow",
                    )
                payload["url"] = None
            return payload
        _fail(f"AMP answered without an evaluation id ({response.status_code}).")
    _refused(response, f"run {execution_id}")
    raise AssertionError("unreachable")


def _wait(client: PlusAPI, evaluation_id: str, url: str | None) -> dict[str, Any]:
    """Poll until the evaluation is done or failed; Ctrl-C leaves it running."""
    console.print("Waiting for the verdict…", style="dim")
    where = f" at {url}" if url else ""
    subject = f"evaluation {evaluation_id}"
    misses = (
        0  # AMP unreachable or answering 5xx: a blip is retried, a streak is reported
    )
    try:
        while True:
            try:
                response = client.get_evaluation(evaluation_id)
            except httpx.HTTPError as error:
                misses += 1
                if misses >= POLL_RETRIES:
                    _fail(
                        f"Could not reach AMP while waiting ({error}); the evaluation keeps running{where}."
                    )
                time.sleep(POLL_SECONDS)
                continue
            if response.status_code >= 500:
                misses += 1
                if misses >= POLL_RETRIES:
                    _refused(response, subject)
                time.sleep(POLL_SECONDS)
                continue
            if response.status_code != 200:
                _refused(response, subject)
            misses = 0
            payload = _payload(response) or {}
            status = payload.get("status")
            if status == "done" and not _well_formed_verdict(payload.get("verdict")):
                _fail(
                    f"AMP answered done without a verdict (protocol error); follow it{where or ' on AMP'}."
                )
            if status in FINISHED:
                return payload
            if status not in STATUSES:
                _fail(
                    f"AMP answered without a known evaluation status ({status!r}); follow it{where or ' on AMP'}."
                )
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        console.print(Text(f"\nStill running{where}."), style="yellow")
        raise SystemExit(130) from None


def _well_formed_verdict(verdict: Any) -> bool:
    """`{"gate": "<word>", "grades": {area: 1..5 | null}}` — anything else is a
    protocol error. A grade is an exact integer in range: `True` is an `int` to
    Python and 6 is not a grade, and neither may print as one."""
    return (
        isinstance(verdict, dict)
        and isinstance(verdict.get("gate"), str)
        and isinstance(verdict.get("grades"), dict)
        and all(_a_grade(grade) for grade in verdict["grades"].values())
    )


def _a_grade(grade: Any) -> bool:
    return grade is None or (type(grade) is int and 1 <= grade <= 5)


def _print_verdict(finished: dict[str, Any], url: str | None) -> None:
    """Everything printed here came over the wire, so it is composed as `Text`
    and never as markup: a `Console` parses square brackets, and an area named
    `[red]tasks[/red]`, a gate, an error or a URL carrying one would restyle
    the line or break it. A fixed list of areas used to make that impossible;
    printing what arrives does not, so the escaping is explicit instead."""
    if finished.get("status") != "done":
        console.print(
            Text(f"Evaluation failed: {finished.get('error') or 'no reason given'}"),
            style="bold red",
        )
        return
    verdict = finished["verdict"]  # _wait let only a well-formed one through
    gate = str(verdict["gate"]).upper()
    style = {"PASSED": "bold green", "FAILED": "bold red"}.get(gate, "bold yellow")
    line = Text("Goal gate: ")
    line.append(gate, style=style)
    # Whatever areas the evaluation graded, in the order it sent them — never a
    # fixed list. The areas are the evaluator's to name, and a client that
    # printed its own would silently drop any it had not heard of while
    # inventing "not measured" for ones that no longer exist. An evaluation
    # that graded nothing prints the gate alone: the separator belongs to the
    # segment after it, so there is never one with nothing behind it.
    for area, grade in (verdict.get("grades") or {}).items():
        line.append(" · ")
        line.append(
            f"{area} {grade}/5" if grade is not None else f"{area} not measured"
        )
    console.print(line)
    if url:
        console.print(Text(f"Full report: {url}"))


def _open(url: str) -> None:
    if is_dmn_mode_enabled():
        return
    with contextlib.suppress(Exception):  # no browser is not an error
        webbrowser.open(url)


def _payload(response: httpx.Response) -> dict[str, Any] | None:
    try:
        loaded = response.json()
    except ValueError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _refused(response: httpx.Response, subject: str) -> None:
    """AMP's own words when it sent them, then exit 1. SUBJECT is "run <id>" or "evaluation <id>"."""
    payload = _payload(response) or {}
    message = str(payload.get("message") or "").strip()
    error = str(payload.get("error") or "")
    if response.status_code in (401, 403):
        if error == "account_required" and message:
            _fail(message)
        _fail(
            f"{message or 'AMP refused the credential'}. Log in with `crewai login` and try again."
        )
    if response.status_code == 404:
        _fail(message or f"AMP answered 404 for {subject}.")
    if response.status_code == 429:
        retry = response.headers.get("Retry-After")
        _fail(
            f"{message or 'AMP is rate limiting this request'}{f' — retry after {retry}s' if retry else ''}."
        )
    _fail(f"AMP answered {response.status_code}{': ' + message if message else ''}.")


def _fail(message: str) -> None:
    # `Text`, because most of what reaches here is AMP's own sentence and a
    # `Console` parses square brackets. No caller relies on markup; the colour
    # comes from `style`.
    console.print(Text(message), style="bold red")
    raise SystemExit(1)
