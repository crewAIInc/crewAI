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

`crewai eval --models "a,b"` compares models instead: AMP finds the project's
deployment by `[tool.crewai].project_id` (or `--deployment`), which runs once as
deployed and once per model; each run is graded, and the comparison — the four
grades, cost and time per model, and what would make the agents do better — is
printed when it is done. It always needs the login: it runs the deployment.
Every evaluation, of either kind, is filed under the project id.
"""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from datetime import datetime, timedelta, timezone
from ipaddress import ip_address
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, NoReturn, cast
from urllib.parse import urlparse
import uuid
import webbrowser

import click
from crewai_core.constants import DEFAULT_CREWAI_ENTERPRISE_URL
from crewai_core.settings import Settings
from dotenv import load_dotenv, set_key
import httpx
from rich.console import Console
from rich.table import Table
from rich.text import Text

from crewai_cli.authentication.token import AuthError, get_auth_token
from crewai_cli.plus_api import PlusAPI
from crewai_cli.utils import (
    get_or_create_project_id,
    get_project_id,
    is_dmn_mode_enabled,
)


console = Console()

LAST_RUN_FILE = Path(".crewai") / "last_run.json"
TRACING_ENV_VAR = "CREWAI_TRACING_ENABLED"
POLL_SECONDS = 3.0
POLL_RETRIES = 5  # consecutive unreachable / 5xx polls before giving up; the evaluation keeps running

# A run's spans reach AMP a little after the run ends — the exporter sends them
# as the process closes and AMP has its own queue behind that. A command that
# just watched the run would otherwise ask for a trace that is still in flight
# and be told, correctly and uselessly, that there is nothing there. So a run
# we know is fresh gets a wait; an id somebody typed does not, and fails at
# once as it always has.
SPANS_WAIT_SECONDS = 120.0
SPANS_POLL_SECONDS = 5.0
RUN_IS_FRESH_SECONDS = 600.0
FINISHED = {"done", "failed"}
STATUSES = {"queued", "running"} | FINISHED


def _record_usage(*, logged_in: bool) -> None:
    """Count an evaluation that is actually starting, and whether the caller was
    logged in.

    Usage stats are anonymous, so nothing that names the run or the account is
    sent — no execution id, no organization, and nothing about the run's content.
    Which runs were evaluated, and by whom, is AMP's record, not these stats'.

    The TUI's button counts `cli_usage:evaluate` when it is pressed, so the
    difference between that and `cli_usage:eval` is intent that never became an
    evaluation.
    """
    try:
        from crewai_core.telemetry import Telemetry

        telemetry = Telemetry()
        telemetry.set_tracer()
        telemetry.feature_usage_span(
            "cli_usage:eval",
            {"authenticated": "true" if logged_in else "false"},
        )
    except Exception:  # noqa: S110 - telemetry must never break a command
        pass


NOT_TRACED = (
    "The run finished but no trace was recorded: the run may have failed, sharing the "
    "trace was declined, or this project's crewai is older than the version that records "
    f"the last run ({LAST_RUN_FILE}). Run the crew again and accept when asked, then `crewai eval`."
)


class EvaluationStoppedError(RuntimeError):
    """The evaluation cannot go on, in words meant for a reader.

    Raised rather than printed-and-exited, because the same two functions serve
    the terminal and the run app: one of them owns the screen, and a line
    printed underneath it is a smear nobody asked for.
    """


def _note(text: str, style: str = "dim") -> None:
    console.print(Text(text), style=style)


def eval_crew(run_id: str | None = None) -> None:
    """Evaluate the last traced run of this project, or the run RUN_ID."""
    project_id = get_or_create_project_id()
    # Read before the project's .env is loaded, so a project cannot add itself.
    trusted = _trusted_amp_origins()
    _load_project_env()
    record = read_last_run() or {}
    execution_id = run_id or record.get("execution_id")
    if execution_id is None:
        # Nothing traced here: the crew runs first, and the app that runs it
        # carries the evaluation on its own screen — link, progress, verdict.
        # It comes back with the run to grade when the app did NOT get to it: a
        # conversational session, or a flow that took the terminal instead.
        execution_id = _run_and_let_the_app_evaluate()
        if execution_id is None:
            return
        record = read_last_run() or {}

    try:
        client = _amp_client(trusted)
    except EvaluationStoppedError as stopped:
        _fail(str(stopped))
    recorded_amp = str(record.get("amp_base_url") or "").rstrip("/")
    if not run_id and recorded_amp and recorded_amp != client.base_url.rstrip("/"):
        console.print(
            Text(
                f"The run was traced to {recorded_amp}; evaluating at the configured AMP {client.base_url}."
            ),
            style="yellow",
        )
    try:
        started = _start_evaluation(
            client,
            execution_id,
            wait_for_spans=run_id is None and _ran_just_now(record),
            project_id=project_id,
        )
    except EvaluationStoppedError as stopped:
        _fail(str(stopped))
    # After, not before: `cli_usage:eval` counts an evaluation, and a refused
    # request — a run AMP does not hold, a credential it will not take — is not
    # one. `_start_evaluation` raises rather than returning on those.
    _record_usage(logged_in=client.api_key is not None)
    url = started.get("url")
    console.print(Text("Evaluating run ").append(execution_id, style="bold"))
    if url:
        # Appended, never interpolated: this line invites a click, so a `url`
        # carrying `[link=…]` would print a trustworthy label over a hostile
        # target. The style belongs to the span, not to the string.
        console.print(Text("Follow it at ").append(url, style="cyan underline"))
        _open(url)

    console.print("Waiting for the verdict…", style="dim")
    try:
        finished = _wait(client, started["id"], url)
    except EvaluationStoppedError as stopped:
        _fail(str(stopped))
    except KeyboardInterrupt:
        console.print(
            Text(f"\nStill running{f' at {url}' if url else ''}."), style="yellow"
        )
        raise SystemExit(130) from None
    _print_verdict(finished, url)
    _say_where_the_criteria_live(write_eval_config(finished))
    # The exit code is what a CI job reads, so it is the gate's: 0 only for a
    # run that PASSED. A failed gate, one without a verdict, and an evaluation
    # that stopped are all 1 — a pipeline that carried on past any of them would
    # ship what the evaluation did not vouch for.
    if not _gate_passed(finished):
        raise SystemExit(1)


def _gate_passed(finished: dict[str, Any]) -> bool:
    verdict = finished.get("verdict") if finished.get("status") == "done" else None
    return isinstance(verdict, dict) and str(verdict.get("gate")).lower() == "passed"


def _ran_just_now(record: dict[str, Any]) -> bool:
    """Did the project record this run in the last few minutes?

    The one thing that tells a run still landing in AMP apart from a run AMP
    does not hold: when it finished. A record without a readable stamp is not
    assumed fresh — the wait is for the case we can name.
    """
    stamp = str(record.get("recorded_at") or record.get("finished_at") or "")
    if not stamp:
        return False

    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return False

    now = datetime.now(when.tzinfo) if when.tzinfo else datetime.now()
    return 0 <= (now - when).total_seconds() <= RUN_IS_FRESH_SECONDS


EVAL_CONFIG_FILE = "eval.jsonc"
# A project's criteria are criteria, not a corpus — the same bound AMP holds
# the field to, checked here so a file that will be refused is not uploaded.
MAX_EVAL_CONFIG_BYTES = 64 * 1024


def project_eval_config(*, note: Callable[[str], None] = _note) -> str | None:
    """What this project says good means, if it has said.

    Read from the project's own directory, beside `pyproject.toml`, because
    that is where the team keeps the things they argue about in review. Not
    parsed here: what a criterion means is the grader's to say, and a document
    this CLI could not read is one the grader will refuse with the line that
    is wrong.
    """
    path = Path.cwd() / EVAL_CONFIG_FILE
    try:
        if not path.is_file():
            return None
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    if len(text.encode("utf-8")) > MAX_EVAL_CONFIG_BYTES:
        # Through NOTE, which the run app routes onto its own screen: a print
        # would land under its layout.
        note(
            f"{EVAL_CONFIG_FILE} is larger than "
            f"{MAX_EVAL_CONFIG_BYTES // 1024}KB and was not sent; this run is graded on "
            "the crew's own expectations."
        )
        return None
    return text or None


def write_eval_config(finished: dict[str, Any]) -> Path | None:
    """Write the criteria this run was graded on, for the project to edit.

    Only when the project has none: after that the file is theirs, and the one
    thing worse than no criteria is criteria that get overwritten every time
    somebody runs an evaluation.
    """
    text = finished.get("eval_config")
    if not isinstance(text, str) or not text.strip():
        return None

    path = Path.cwd() / EVAL_CONFIG_FILE
    try:
        if path.exists() or not path.parent.is_dir():
            return None
        path.write_text(text, encoding="utf-8")
    except OSError:
        return None
    return path


EVAL_MARKER_FILE = "last_eval.json"


def _marker_path() -> Path:
    from crewai.telemetry.tracing.last_run import LAST_RUN_DIR, project_dir

    return project_dir() / LAST_RUN_DIR / EVAL_MARKER_FILE


def record_evaluation_outcome(execution_id: str | None) -> None:
    """What the run app did with the evaluation it was opened for.

    The app runs in a child process for most projects, so nothing in memory
    reaches the command waiting behind it. Two things that command must know
    are both here: that this run has been evaluated already, and — when the app
    had nothing to evaluate — that it was the app saying so rather than no app
    at all. A conversational session and a flow that takes the terminal never
    get here, leave no marker, and the command grades the run itself.

    Never raises: a marker that cannot be written costs a second evaluation,
    not the run.
    """
    with contextlib.suppress(Exception):
        path = _marker_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "execution_id": execution_id,
                    "at": datetime.now(timezone.utc).isoformat(),
                }
            ),
            encoding="utf-8",
        )


def evaluation_marker(after: datetime) -> dict[str, Any] | None:
    """The app's word about the run this command just watched, if it left one.

    AFTER is when the run was started, and the marker must be newer: a project
    keeps one marker, and the one from an earlier `crewai eval` is not about
    this run. Matched on time rather than on the project's last-run record,
    which another run finishing in the same project can replace while this app
    is still open.

    A second of slack, because the two stamps are written by two processes and
    the question being asked is "this run or an older one", which a second
    cannot confuse.
    """
    with contextlib.suppress(Exception):
        marker = json.loads(_marker_path().read_text(encoding="utf-8"))
        when = datetime.fromisoformat(str(marker.get("at")))
        if isinstance(marker, dict) and when >= after - timedelta(seconds=1):
            return dict(marker)
    return None


def evaluate_run(
    execution_id: str,
    *,
    on_started: Callable[[dict[str, Any]], None],
    on_status: Callable[[dict[str, Any]], None] | None = None,
    note: Callable[[str], None] = _note,
) -> dict[str, Any]:
    """Start the evaluation of EXECUTION_ID and wait for its verdict.

    The run app's door into this command: it has a screen of its own to show
    the link and the verdict on, so nothing here prints and nothing exits — the
    url reaches ON_STARTED as soon as AMP gives it, every unfinished answer
    reaches ON_STATUS, and anything that stops the evaluation is an
    `EvaluationStoppedError` carrying the sentence to show.

    The spans of a run that just ended may still be in flight, which is the
    whole reason this is called from inside the app that ran it, so the wait
    for them is always on here.
    """
    client = _amp_client(_machine_amp_origins(), note=note)
    started = _start_evaluation(
        client,
        execution_id,
        wait_for_spans=True,
        note=note,
        project_id=get_project_id(),
    )
    # After the start, exactly as the command counts it: `cli_usage:eval` is the
    # count of evaluations that began, and an evaluation the app runs is one.
    _record_usage(logged_in=client.api_key is not None)
    record_evaluation_outcome(execution_id)
    on_started(started)
    finished = _wait(client, started["id"], started.get("url"), on_status=on_status)
    written = write_eval_config(finished)
    if written is not None:
        finished = {**finished, "wrote_eval_config": written.name}
    return finished


def _say_where_the_criteria_live(path: Path | None) -> None:
    """One line, once: the file exists now, and editing it changes the next grade."""
    if path is None:
        return

    console.print(
        Text(
            f"Wrote {path.name} — say what good means for this crew there, and the next "
            "`crewai eval` is graded on it."
        ),
        style="green",
    )


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


def _machine_amp_origins() -> set[str]:
    """The same set, for a caller whose environment can no longer be trusted.

    `crewai eval` reads `CREWAI_PLUS_URL` BEFORE the project's `.env` is loaded,
    so a shell export counts there and a project cannot introduce one. Inside
    the run app there is no such "before": the crew has already run and its
    `.env` was loaded for it, so the variable is left out of the set entirely.
    The request still follows it — that is how a self-hosted project is wired —
    and the credential still does not.
    """
    candidates = (Settings().enterprise_base_url, DEFAULT_CREWAI_ENTERPRISE_URL)
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


def _amp_client(trusted: set[str], *, note: Callable[[str], None] = _note) -> PlusAPI:
    """The AMP to ask, and whether the saved login goes with it.

    A project's `.env` may point `crewai eval` at another AMP — that is how a
    self-hosted project is wired, and the run was traced there — so the request
    follows it. The credential does not: it goes only to an AMP this machine is
    logged in to, over a connection that encrypts it. Anywhere else the run is
    read anonymously."""
    client = PlusAPI(api_key=saved_login())
    if client.api_key is None:
        return _anonymous(client)

    origin = _origin(client.base_url)
    if origin in trusted and _encrypted(origin):
        return client

    why = (
        "is not an AMP this machine is logged in to"
        if origin not in trusted
        else "would carry the login over plain HTTP"
    )
    note(
        f"Reading anonymously: {client.base_url} {why}. "
        "Run `crewai enterprise configure <url>` to log in to it."
    )
    return _anonymous(PlusAPI())


def _anonymous(client: PlusAPI) -> PlusAPI:
    """A client that says nothing about who is asking.

    `PlusAPI` adds the saved organization to every request, and AMP reads it
    only beside a credential — so without one it tells AMP nothing, and tells
    an AMP this machine is not logged in to which organization is. Dropped, as
    anonymous trace grants drop it.
    """
    client.headers.pop("X-Crewai-Organization-Id", None)
    return client


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
        # Raised, not printed-and-exited: the run app shows this on its own
        # screen, where a print would land under the layout and an exit would
        # take the sentence with it.
        raise EvaluationStoppedError(
            f"Could not read the saved login ({type(error).__name__}: {error}). "
            "Run `crewai login` again, or `crewai eval` will not know who you are."
        ) from error


def _run_and_let_the_app_evaluate() -> str | None:
    """No traced run recorded here: offer to turn tracing on and run the crew.

    The run app evaluates what it ran, on its own screen, so this returns None
    when it did — the reader has already seen the link and the verdict. It
    returns the run to grade when no app got to it (a conversational session, a
    flow that took the terminal), and says so itself when nothing was traced.
    """
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
        # `Text`, never markup: the reason may be an OS error's own words, and
        # its `[Errno 13]` would be read as a style tag.
        console.print(Text(_nothing_traced_unattended() or steps), style="yellow")
        raise SystemExit(1)
    if not click.confirm(
        "No traced run is recorded in this project. Turn tracing on and run the crew now? "
        "The run's trace is sent to CrewAI AMP.",
        default=True,  # João, 2026-09-20: y/n with Y as the default — the prompt says what Enter does
    ):
        console.print(steps, style="yellow")
        raise SystemExit(0)

    _enable_tracing()
    from crewai_cli.crew_run_tui import evaluating_after_run
    from crewai_cli.run_crew import run_crew

    # The app is told an evaluation is waiting and starts it itself when the run
    # ends. It runs in this process for some projects and in a child for others,
    # so what says a run was traced is either the holder's id or the record the
    # run wrote.
    began = datetime.now(timezone.utc)
    with evaluating_after_run() as watched:
        run_crew()

    # The app says what it did with the run it watched, and its word settles
    # this: it evaluated it (nothing left to do), or it had nothing to evaluate
    # (say so once, in the terminal the screen has left). Only when no app got
    # here at all — a conversational session, a flow that took the terminal —
    # is there a run for this command to grade.
    marker = evaluation_marker(after=began)
    if marker is not None:
        if marker.get("execution_id"):
            return None
        console.print(NOT_TRACED, style="bold red")
        raise SystemExit(1)

    # The project's record, only when it was written after this run began: a
    # record is the project's LAST run, and another run finishing in the same
    # project would otherwise be graded in this one's place.
    record = read_last_run() or {}
    traced = watched["execution_id"] or (
        record.get("execution_id") if _recorded_since(record, began) else None
    )
    if traced:
        return str(traced)

    console.print(NOT_TRACED, style="bold red")
    raise SystemExit(1)


def _recorded_since(record: dict[str, Any], began: datetime) -> bool:
    """Was RECORD written after BEGAN? A record without a readable, zoned stamp
    is not assumed to be this run's."""
    stamp = str(record.get("recorded_at") or record.get("finished_at") or "")
    try:
        when = datetime.fromisoformat(stamp)
    except ValueError:
        return False
    if when.tzinfo is None:
        return False
    return when >= began - timedelta(seconds=1)


def _nothing_traced_unattended() -> str | None:
    """Why a run with tracing on left nothing to evaluate, when nobody was there.

    An anonymous run asks before its trace leaves the machine, and a process with
    no terminal has nobody to ask — so its trace is kept local, tracing on or
    not. Telling that user to turn tracing on sends them round the same loop;
    logging in is what makes an unattended run traced. None when tracing is off
    or there is a login: the ordinary steps are the right ones then. A login
    that cannot be read says so instead.
    """
    if os.environ.get(TRACING_ENV_VAR, "").strip().lower() not in ("true", "1"):
        return None
    try:
        if saved_login() is not None:
            return None
    except EvaluationStoppedError as unreadable:
        # A login that exists and cannot be read is the reason, and its
        # sentence says what to do about it.
        return str(unreadable)
    return (
        "No traced run is recorded in this project. Tracing is on, but a run nobody is "
        "watching is only traced when you are logged in: run `crewai login`, then "
        "`crewai run` and `crewai eval` again."
    )


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


def _start_evaluation(
    client: PlusAPI,
    execution_id: str,
    *,
    wait_for_spans: bool = False,
    note: Callable[[str], None] = _note,
    project_id: str | None = None,
) -> dict[str, Any]:
    deadline = time.monotonic() + SPANS_WAIT_SECONDS if wait_for_spans else 0.0
    said = False
    # Read once: the file does not change while the spans are in flight, and a
    # warning about it belongs on the screen once, not on every retry.
    eval_config = project_eval_config(note=note)
    while True:
        try:
            response = client.create_evaluation(
                execution_id, eval_config=eval_config, project_id=project_id
            )
        except httpx.HTTPError as error:
            raise EvaluationStoppedError(
                f"Could not reach AMP to start the evaluation: {error}"
            ) from error
        # The run finished moments ago and its spans are still on their way:
        # waiting is the answer, not a 404 the reader can do nothing with.
        if response.status_code == 404 and time.monotonic() < deadline:
            if not said:
                note(
                    "The run's trace has not reached CrewAI AMP yet — waiting up to "
                    f"{int(SPANS_WAIT_SECONDS // 60)} minutes for it…"
                )
                said = True
            time.sleep(SPANS_POLL_SECONDS)
            continue
        break
    return _accepted(response, f"run {execution_id}", note=note)


def _accepted(
    response: httpx.Response,
    subject: str,
    *,
    note: Callable[[str], None] = _note,
    about_a_deployment: bool = False,
) -> dict[str, Any]:
    """AMP's answer to a start: the evaluation it opened, or its refusal as a sentence."""
    if response.status_code in (200, 202):
        payload = _payload(response)
        # The id is what every later call is made with, so a missing or
        # non-string one is a protocol error and not something to carry on
        # with. The url is only ever shown and opened, so a malformed one
        # costs the link and nothing else: the evaluation is already running
        # and its verdict is what the user came for.
        if payload and isinstance(payload.get("id"), str) and payload["id"]:
            url = _report_url(payload.get("url"))
            if url is None and payload.get("url") is not None:
                note(
                    "AMP answered with a report url that is not a web address this "
                    "command will open; the link is unavailable for this run."
                )
            payload["url"] = url
            return payload
        raise EvaluationStoppedError(
            f"AMP answered without an evaluation id ({response.status_code})."
        )
    raise EvaluationStoppedError(
        _refusal_message(response, subject, about_a_deployment=about_a_deployment)
    )


def _report_url(value: Any) -> str | None:
    """The report link, if it is one a browser may be sent to.

    It is printed as a link and opened without a click, and it comes from
    whichever AMP answered — which a project's `.env` may choose. So it must be
    an absolute web address: HTTPS, or plain HTTP to this machine, with no
    credentials in it and no control characters to restyle the terminal. The
    host is not pinned: the report is served by the evaluation service, not by
    AMP, and a self-hosted AMP names its own.
    """
    if (
        not isinstance(value, str)
        or not value
        or any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        return None
    try:
        parsed = urlparse(value)
        if (
            parsed.hostname
            and parsed.username is None
            and parsed.password is None
            and _encrypted(f"{parsed.scheme}://{parsed.netloc}")
        ):
            return value
    except ValueError:
        pass
    return None


def _wait(
    client: PlusAPI,
    evaluation_id: str,
    url: str | None,
    *,
    on_status: Callable[[dict[str, Any]], None] | None = None,
    answer: str = "verdict",
) -> dict[str, Any]:
    """Poll until the evaluation is done or failed.

    Every unfinished answer goes to ON_STATUS, so a caller that has somewhere to
    show progress can show it; the caller decides what a Ctrl-C means. A done
    answer must carry a well-formed ANSWER: a run's `verdict`, or a models
    evaluation's `comparison`.
    """
    well_formed = _WELL_FORMED[answer]
    where = f" at {url}" if url else ""
    subject = f"evaluation {evaluation_id}"
    misses = (
        0  # AMP unreachable or answering 5xx: a blip is retried, a streak is reported
    )
    while True:
        try:
            response = client.get_evaluation(evaluation_id)
        except httpx.HTTPError as error:
            misses += 1
            if misses >= POLL_RETRIES:
                raise EvaluationStoppedError(
                    f"Could not reach AMP while waiting ({error}); the evaluation keeps running{where}."
                ) from error
            time.sleep(POLL_SECONDS)
            continue
        if response.status_code >= 500:
            misses += 1
            if misses >= POLL_RETRIES:
                raise EvaluationStoppedError(_refusal_message(response, subject))
            time.sleep(POLL_SECONDS)
            continue
        if response.status_code != 200:
            raise EvaluationStoppedError(_refusal_message(response, subject))
        misses = 0
        payload = _payload(response) or {}
        status = payload.get("status")
        if status == "done" and not well_formed(payload):
            raise EvaluationStoppedError(
                f"AMP answered done without a {answer} (protocol error); follow it{where or ' on AMP'}."
            )
        if status in FINISHED:
            return payload
        if status not in STATUSES:
            raise EvaluationStoppedError(
                f"AMP answered without a known evaluation status ({status!r}); follow it{where or ' on AMP'}."
            )
        if on_status is not None:
            on_status(payload)
        time.sleep(POLL_SECONDS)


def _carries_a_verdict(payload: dict[str, Any]) -> bool:
    return _well_formed_verdict(payload.get("verdict"))


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


def _carries_a_comparison(payload: dict[str, Any]) -> bool:
    """`{"comparison": {"models": [{...}, ...]}}` — the rows are read cell by
    cell when printed, and a cell that is not what it should be prints as "—";
    a comparison with no rows to print is a protocol error."""
    comparison = payload.get("comparison")
    return (
        isinstance(comparison, dict)
        and isinstance(comparison.get("models"), list)
        and bool(comparison["models"])
        and all(isinstance(row, dict) for row in comparison["models"])
    )


_WELL_FORMED: dict[str, Callable[[dict[str, Any]], bool]] = {
    "verdict": _carries_a_verdict,
    "comparison": _carries_a_comparison,
}


# ── crewai eval --models ─────────────────────────────────────────────────────

MAX_MODELS = 5
MAX_MODEL_CHARS = 200
MODELS_EXAMPLE = '--models "openai/gpt-4o-mini,anthropic/claude-haiku-4-5"'
LOGIN_REQUIRED = (
    "Comparing models runs your deployment, so it needs your CrewAI AMP account: "
    "log in with `crewai login` and run it again."
)
GRADE_COLUMNS = ("goal", "tasks", "agents", "tools")
TOP_SUGGESTIONS = 3


def parse_models(text: str | None) -> list[str]:
    """The models to compare, from ONE comma-separated list.

    Items are stripped and a repeat is dropped. Each must name its provider
    (`provider/model`) — the deployment builds the model from exactly that
    string, and a bare `gpt-4o` would be a guess about which provider bills it.
    """
    models: list[str] = []
    for raw in (text or "").split(","):
        item = raw.strip()
        if not item or item in models:
            continue
        provider, separator, name = item.partition("/")
        if not (separator and provider.strip() and name.strip()):
            raise EvaluationStoppedError(
                f"{item!r} names no provider: write each model as provider/model, "
                f"e.g. {MODELS_EXAMPLE}."
            )
        if len(item) > MAX_MODEL_CHARS:
            raise EvaluationStoppedError(
                f"{item[:40]!r}… is longer than {MAX_MODEL_CHARS} characters; "
                "that is not a model name."
            )
        models.append(item)
    if not models:
        raise EvaluationStoppedError(
            f"--models names no model: give one to {MAX_MODELS}, e.g. {MODELS_EXAMPLE}."
        )
    if len(models) > MAX_MODELS:
        raise EvaluationStoppedError(
            f"--models names {len(models)} models; compare at most {MAX_MODELS} at once."
        )
    return models


def _deployment_id(value: str | None) -> str | None:
    """`--deployment` as the UUID AMP knows the deployment by, or None."""
    if value is None:
        return None
    try:
        return str(uuid.UUID(value.strip()))
    except ValueError:
        raise EvaluationStoppedError(
            f"--deployment {value!r} is not a deployment id; it is the UUID AMP "
            "shows for the deployment."
        ) from None


# Model names that are the same for everyone who types them are public. What
# else a `--models` item can carry is a customer's: a fine-tune id
# (`openai/ft:gpt-4o-mini:acme-corp::abc`), an Azure deployment name, a
# self-hosted model or host. Those are sent as `<provider>/other`.
OTHER_MODEL = "other"
# The vendor segment of an aggregator id (`openrouter/openai/gpt-4o-mini`)
# that is itself public.
_PUBLIC_VENDORS = frozenset(
    {"openai", "anthropic", "google", "meta-llama", "mistralai", "deepseek", "qwen"}
)


def telemetry_model_name(model: str) -> str:
    """MODEL as the usage stats may carry it: as typed when it is a model crewAI
    knows, else `<provider>/other` — and `other/other` for a provider crewAI does
    not know, since a provider string can name a host too.

    Known means an exact entry of crewAI's model catalog (the context-window
    tables every provider resolves against), never a prefix of one: a fine-tune
    or a deployment named after a public model is still the customer's.
    """
    catalog, providers = _known_models_and_providers()
    provider, _, name = model.partition("/")
    if provider not in providers:
        return f"{OTHER_MODEL}/{OTHER_MODEL}"
    vendor, nested, tail = name.partition("/")
    public = (
        name in catalog
        or model in catalog
        or (nested and vendor in _PUBLIC_VENDORS and tail in catalog)
    )
    if not public or any(part.startswith("ft:") for part in model.split("/")):
        return f"{provider}/{OTHER_MODEL}"
    return model


def _known_models_and_providers() -> tuple[frozenset[str], frozenset[str]]:
    """crewAI's catalog of models and the providers it routes; empty when this
    environment's crewai cannot say, so every model is then sent as "other"."""
    from crewai_cli.constants import PROVIDERS

    try:
        from crewai.llm import SUPPORTED_NATIVE_PROVIDERS
        from crewai.llms.context_window import LLM_CONTEXT_WINDOW_SIZES
    except Exception:
        return frozenset(), frozenset()
    return (
        frozenset(LLM_CONTEXT_WINDOW_SIZES),
        frozenset(SUPPORTED_NATIVE_PROVIDERS) | frozenset(PROVIDERS),
    )


def _record_models_usage(models: list[str]) -> None:
    """Count a comparison that is actually starting, and which models it compares.

    The models go through `telemetry_model_name`: a public model by name, any
    other as `<provider>/other`. Nothing names the run, the deployment or the
    organization. Always logged in: a comparison cannot start without the
    account.
    """
    try:
        from crewai_core.telemetry import Telemetry

        telemetry = Telemetry()
        telemetry.set_tracer()
        telemetry.feature_usage_span(
            "cli_usage:eval_models",
            {
                "authenticated": "true",
                "models": ",".join(telemetry_model_name(m) for m in models),
                "models_count": str(len(models)),
            },
        )
    except Exception:  # noqa: S110 - telemetry must never break a command
        pass


def eval_models(models_text: str, deployment_id: str | None = None) -> None:
    """Run this project's deployment once as deployed and once per model, and compare.

    The deployment is AMP's to find, by the project id, unless DEPLOYMENT_ID
    names one; its own models are the baseline, read off the deployment rather
    than this checkout, which may differ from what was deployed.
    """
    try:
        models = parse_models(models_text)
        deployment = _deployment_id(deployment_id)
    except EvaluationStoppedError as stopped:
        _fail(str(stopped))
    if not Path("pyproject.toml").is_file():
        _fail(
            "No crewAI project here (no pyproject.toml). Run `crewai eval --models` "
            "from the directory of the project you deployed."
        )
    project_id = get_or_create_project_id()
    if not project_id:
        _fail(
            "Could not read or write [tool.crewai].project_id in pyproject.toml, which "
            "is how AMP finds this project's deployment."
        )
    # Read before the project's .env is loaded, so a project cannot add itself.
    trusted = _trusted_amp_origins()
    _load_project_env()
    try:
        client = _amp_client(trusted)
    except EvaluationStoppedError as stopped:
        _fail(str(stopped))
    if client.api_key is None:
        _fail(LOGIN_REQUIRED)

    try:
        response = client.create_models_evaluation(
            models,
            project_id=project_id,
            eval_config=project_eval_config(),
            deployment_id=deployment,
        )
    except httpx.HTTPError as error:
        _fail(f"Could not reach AMP to start the comparison: {error}")
    try:
        started = _accepted(response, "the comparison", about_a_deployment=True)
    except EvaluationStoppedError as stopped:
        _fail(str(stopped))
    # After, not before, as `cli_usage:eval` is counted: a refused request —
    # no deployment, one this account may not run — is not a comparison.
    _record_models_usage(models)
    url = started.get("url")
    console.print(
        Text("Comparing ")
        .append(", ".join(models), style="bold")
        .append(" with the deployed models")
    )
    if url:
        # Appended, never interpolated — the same reason as `eval_crew`'s link.
        console.print(Text("Follow it at ").append(url, style="cyan underline"))
        _open(url)

    console.print("Waiting for the comparison…", style="dim")
    shown: list[str] = []

    def show_progress(payload: dict[str, Any]) -> None:
        line = _progress_line(payload)
        if line and (not shown or shown[-1] != line):
            shown.append(line)
            _note(line)

    try:
        finished = _wait(
            client, started["id"], url, on_status=show_progress, answer="comparison"
        )
    except EvaluationStoppedError as stopped:
        _fail(str(stopped))
    except KeyboardInterrupt:
        console.print(
            Text(f"\nStill running{f' at {url}' if url else ''}."), style="yellow"
        )
        raise SystemExit(130) from None
    _print_comparison(finished, url)
    # The criteria the comparison was graded on, for the project to edit — only
    # when it has none, exactly as after a Mode 1 evaluation.
    _say_where_the_criteria_live(write_eval_config(finished))
    if finished.get("status") != "done":
        raise SystemExit(1)


def _progress_line(payload: dict[str, Any]) -> str | None:
    """One line for where the comparison is, from the poll's `progress` (or the
    last of its `events`): which model is running, "model 2 of 3", and the
    subject being judged. What is not there is left out, never guessed."""
    progress = payload.get("progress")
    events = payload.get("events")
    if progress is None and isinstance(events, list) and events:
        progress = events[-1]
    if isinstance(progress, str):
        return progress.strip() or None
    if not isinstance(progress, dict):
        return None
    if isinstance(progress.get("message"), str) and progress["message"].strip():
        return str(progress["message"]).strip()
    event = progress.get("event")
    detail = progress.get("payload")
    if not isinstance(detail, dict):
        detail = progress
    index, total = detail.get("index"), detail.get("total")
    position = (
        f"model {index + 1} of {total}"
        if type(index) is int and type(total) is int and 0 <= index < total
        else None
    )
    name = next(
        (
            str(detail[key])
            for key in ("label", "key")
            if isinstance(detail.get(key), str) and detail[key]
        ),
        None,
    )
    subject = detail.get("subject")
    if event == "judging" or isinstance(subject, str):
        parts = [f"judging {subject}" if isinstance(subject, str) else "judging"]
    elif event == "configuration_done":
        parts = ["graded"]
    else:
        parts = ["running"]
    parts += [part for part in (position, name) if part]
    known = event is not None or isinstance(subject, str) or len(parts) > 1
    return " · ".join(parts) if known else None


def _print_comparison(finished: dict[str, Any], url: str | None) -> None:
    """The models side by side, then what would make them better.

    Every cell came over the wire, so each is a `Text`: a label such as
    `Writer: [red]x[/red]` prints as written. The baseline — the deployment as
    it is — is marked, and the best value in each column is starred: the highest
    grade, the lowest cost and time. A column where every model is the same, or
    only one has a value, stars nothing: there is no "best" to point at.
    """
    if finished.get("status") != "done":
        console.print(
            Text(f"Comparison failed: {finished.get('error') or 'no reason given'}"),
            style="bold red",
        )
        if url:
            console.print(Text(f"Report: {url}"))
        return
    comparison = finished["comparison"]  # _wait let only a well-formed one through
    rows: list[dict[str, Any]] = comparison["models"]

    grades = {area: [_grade_of(row, area) for row in rows] for area in GRADE_COLUMNS}
    costs = [_number(row.get("cost_usd")) for row in rows]
    seconds = [_number(row.get("seconds")) for row in rows]
    best = {area: _best(values, max) for area, values in grades.items()}
    cheapest, fastest = _best(costs, min), _best(seconds, min)

    table = Table(show_edge=False, pad_edge=False)
    for header in ("model", *GRADE_COLUMNS, "cost", "time"):
        table.add_column(header, justify="left" if header == "model" else "right")
    for n, row in enumerate(rows):
        label = Text(str(row.get("label") or row.get("key") or "?"))
        if row.get("baseline") is True:
            label.append(" (deployed)", style="dim")
        cells = [label]
        for area in GRADE_COLUMNS:
            grade = grades[area][n]
            cells.append(
                _starred(f"{grade}/5" if grade is not None else "—", grade, best[area])
            )
        cost, took = costs[n], seconds[n]
        cells.append(
            _starred(f"${cost:.4f}" if cost is not None else "—", cost, cheapest)
        )
        cells.append(
            _starred(f"{took:.1f}s" if took is not None else "—", took, fastest)
        )
        table.add_row(*cells)
    console.print(table)

    suggestions = _top_suggestions(comparison.get("suggestions"))
    if suggestions:
        console.print(Text("What would make it better", style="bold"))
    for n, item in enumerate(suggestions, 1):
        where = " — ".join(
            str(item[key])
            for key in ("subject", "field")
            if isinstance(item.get(key), str)
        )
        line = Text(f"{n}. ")
        if where:
            line.append(where, style="bold").append(": ")
        line.append(str(item.get("problem") or ""))
        if item.get("shared") is True:
            line.append(" (every model)", style="dim")
        console.print(line)
        if isinstance(item.get("change"), str) and item["change"].strip():
            console.print(Text(f"   change: {item['change'].strip()}"))
    if url:
        console.print(Text(f"Full report: {url}"))


def _grade_of(row: dict[str, Any], area: str) -> int | None:
    grades = row.get("grades")
    grade = grades.get(area) if isinstance(grades, dict) else None
    return grade if grade is not None and _a_grade(grade) else None


def _number(value: Any) -> float | None:
    """A non-negative number, or None: `True` is an `int` to Python and not a cost."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value >= 0 else None


def _best(values: list[Any], pick: Callable[[list[Any]], Any]) -> float | int | None:
    known = [value for value in values if value is not None]
    if len(known) < 2 or len(set(known)) == 1:
        return None
    return cast(float | int, pick(known))


def _starred(text: str, value: Any, best: Any) -> Text:
    cell = Text(text)
    if best is not None and value == best:
        cell.append(" ★", style="yellow")
    return cell


def _top_suggestions(value: Any) -> list[dict[str, Any]]:
    """The first few suggestions, the ones every model needed first — a problem
    every model had points at the prompt, not at a model."""
    if not isinstance(value, list):
        return []
    items = [item for item in value if isinstance(item, dict) and item.get("problem")]
    items.sort(key=lambda item: item.get("shared") is not True)
    return items[:TOP_SUGGESTIONS]


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


def _refusal_message(
    response: httpx.Response, subject: str, *, about_a_deployment: bool = False
) -> str:
    """AMP's own words when it sent them. SUBJECT is "run <id>" or "evaluation <id>".

    A sentence, not a print: the terminal and the run app both show it, and only
    one of them shows it by printing. ABOUT_A_DEPLOYMENT: a 403 that is not about
    the credential — this account may not run that deployment — is AMP's
    sentence alone, because logging in again changes nothing there.
    """
    payload = _payload(response) or {}
    message = str(payload.get("message") or "").strip()
    error = str(payload.get("error") or "")
    if (
        about_a_deployment
        and response.status_code == 403
        and error not in {"bad_credentials", "account_required"}
        and message
    ):
        return message
    if response.status_code in (401, 403):
        if error == "account_required" and message:
            return message
        return f"{message or 'AMP refused the credential'}. Log in with `crewai login` and try again."
    if response.status_code == 404:
        return message or f"AMP answered 404 for {subject}."
    if response.status_code == 429:
        retry = response.headers.get("Retry-After")
        return f"{message or 'AMP is rate limiting this request'}{f' — retry after {retry}s' if retry else ''}."
    return f"AMP answered {response.status_code}{': ' + message if message else ''}."


def _refused(response: httpx.Response, subject: str) -> None:
    """The same words, printed, then exit 1."""
    _fail(_refusal_message(response, subject))


def _fail(message: str) -> NoReturn:
    # `Text`, because most of what reaches here is AMP's own sentence and a
    # `Console` parses square brackets. No caller relies on markup; the colour
    # comes from `style`.
    console.print(Text(message), style="bold red")
    raise SystemExit(1)
