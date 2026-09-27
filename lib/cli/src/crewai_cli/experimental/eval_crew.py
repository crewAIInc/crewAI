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

from collections.abc import Callable
import contextlib
from datetime import datetime, timedelta, timezone
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
    get_or_create_project_id()
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
        )
    except EvaluationStoppedError as stopped:
        _fail(str(stopped))
    # After, not before: `cli_usage:eval` counts an evaluation, and a refused
    # request — a run AMP does not hold, a credential it will not take — is not
    # one. `_start_evaluation` raises rather than returning on those.
    _record_usage(execution_id, logged_in=client.api_key is not None)
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
    if finished.get("status") != "done":
        raise SystemExit(1)


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


def project_eval_config() -> str | None:
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
        console.print(
            Text(
                f"{EVAL_CONFIG_FILE} is larger than "
                f"{MAX_EVAL_CONFIG_BYTES // 1024}KB and was not sent; this run is graded on "
                "the crew's own expectations."
            ),
            style="yellow",
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
    started = _start_evaluation(client, execution_id, wait_for_spans=True, note=note)
    # After the start, exactly as the command counts it: `cli_usage:eval` is the
    # count of evaluations that began, and an evaluation the app runs is one.
    _record_usage(execution_id, logged_in=client.api_key is not None)
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
        return client

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

    traced = watched["execution_id"] or (read_last_run() or {}).get("execution_id")
    if traced:
        return str(traced)

    console.print(NOT_TRACED, style="bold red")
    raise SystemExit(1)


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
) -> dict[str, Any]:
    deadline = time.monotonic() + SPANS_WAIT_SECONDS if wait_for_spans else 0.0
    said = False
    while True:
        try:
            response = client.create_evaluation(
                execution_id, eval_config=project_eval_config()
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
                    note(
                        "AMP answered with a report url that is not a string; "
                        "the link is unavailable for this run."
                    )
                payload["url"] = None
            return payload
        raise EvaluationStoppedError(
            f"AMP answered without an evaluation id ({response.status_code})."
        )
    raise EvaluationStoppedError(_refusal_message(response, f"run {execution_id}"))


def _wait(
    client: PlusAPI,
    evaluation_id: str,
    url: str | None,
    *,
    on_status: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Poll until the evaluation is done or failed.

    Every unfinished answer goes to ON_STATUS, so a caller that has somewhere to
    show progress can show it; the caller decides what a Ctrl-C means.
    """
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
        if status == "done" and not _well_formed_verdict(payload.get("verdict")):
            raise EvaluationStoppedError(
                f"AMP answered done without a verdict (protocol error); follow it{where or ' on AMP'}."
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


def _refusal_message(response: httpx.Response, subject: str) -> str:
    """AMP's own words when it sent them. SUBJECT is "run <id>" or "evaluation <id>".

    A sentence, not a print: the terminal and the run app both show it, and only
    one of them shows it by printing.
    """
    payload = _payload(response) or {}
    message = str(payload.get("message") or "").strip()
    error = str(payload.get("error") or "")
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


def _fail(message: str) -> None:
    # `Text`, because most of what reaches here is AMP's own sentence and a
    # `Console` parses square brackets. No caller relies on markup; the colour
    # comes from `style`.
    console.print(Text(message), style="bold red")
    raise SystemExit(1)
