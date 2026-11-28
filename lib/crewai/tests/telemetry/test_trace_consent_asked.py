"""A trace decision is recorded only when a person was asked.

A run where the question could not be shown (no terminal, a suite under test,
suppressed messages) leaves the machine "not asked", buffers nothing, and the
first run at a terminal still asks. An answer given at a terminal is final.
"""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from functools import partial
import os
from pathlib import Path
import threading
from unittest.mock import Mock
from uuid import uuid4

import appdirs
from crewai.events.listeners.tracing import utils
from crewai.events.listeners.tracing.first_time_trace_handler import (
    FirstTimeTraceHandler,
)
from crewai.flow.flow import Flow, listen, start
from crewai.telemetry.tracing import ephemeral
from crewai.telemetry.tracing.ephemeral import ephemeral_tracing, trace_consent
from crewai.telemetry.tracing.grants import (
    GrantSpanExporter,
    TraceGrant,
    TraceGrantClient,
)
from click.testing import CliRunner
from crewai_cli.cli import traces_enable
from crewai_core import user_data
from crewai_core.runtime_env import CODING_AGENT_ENV_MARKERS, GENERIC_AGENT_ENV_VARS
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import pytest


ASKED_NO = {
    "first_execution_done": True,
    "trace_consent": False,
    "trace_consent_asked": True,
}
ASKED_YES = {
    "first_execution_done": True,
    "trace_consent": True,
    "trace_consent_asked": True,
}
# What versions before `trace_consent_asked` wrote for EVERY first run that
# did not end in a yes, including runs where nobody could be asked.
LEGACY_NO = {"first_execution_done": True, "trace_consent": False}
LEGACY_YES = {"first_execution_done": True, "trace_consent": True}


@pytest.fixture(autouse=True)
def isolated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_STORAGE_DIR", "crewai_trace_consent_asked")
    monkeypatch.setattr(
        "crewai_core.paths.appdirs.user_data_dir",
        lambda app, author: str(tmp_path / app),
    )
    for name in (
        "CREWAI_TRACING_ENABLED",
        "CREWAI_EVAL_AWAITING_RUN",
        "CREWAI_USER_PAT",
        "CREWAI_ENTERPRISE_ACTION_AUTH_TOKEN",
        "CREWAI_EPHEMERAL_TRACE_MAX_SPANS",
        "CREWAI_EPHEMERAL_TRACE_MAX_BYTES",
        "OTEL_SDK_DISABLED",
        *(var for _, env_vars in CODING_AGENT_ENV_MARKERS for var in env_vars),
        *GENERIC_AGENT_ENV_VARS,
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setattr("crewai.telemetry.tracing.grants.get_auth_token", lambda: None)
    # Outside the suite's own marker: whether a terminal is attached is the
    # one thing each test decides.
    monkeypatch.setattr(utils, "_is_test_environment", lambda: False)
    monkeypatch.setattr(utils, "_is_interactive_terminal", lambda: False)


@pytest.fixture
def person_at_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_is_interactive_terminal", lambda: True)


@pytest.fixture
def grants(monkeypatch: pytest.MonkeyPatch) -> list[TraceGrant]:
    """Upload grants AMP issued, without opening a socket."""
    issued: list[TraceGrant] = []

    def create(client: TraceGrantClient, execution_uuid: str) -> TraceGrant:
        grant = TraceGrant(
            token="synthetic-grant",
            collector_url="https://collector.invalid/v1/traces",
            execution_uuid=execution_uuid,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=15),
        )
        issued.append(grant)
        return grant

    monkeypatch.setattr(TraceGrantClient, "create", create)
    monkeypatch.setattr(
        GrantSpanExporter, "_exporter", staticmethod(lambda grant: InMemorySpanExporter())
    )
    return issued


@pytest.fixture
def terminal_answer(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """The terminal prompt, answered without a person typing."""
    prompt = Mock(return_value=False)
    monkeypatch.setattr(ephemeral, "prompt_user_for_trace_viewing", prompt)
    return prompt


def _record(data: dict[str, object]) -> None:
    user_data.update_user_data(data)


def _first_run_traced() -> None:
    with ephemeral_tracing(str(uuid4()), first_time=True) as session:
        session.get_tracer().start_span("execute flow").end()


class ExampleFlow(Flow):
    @start()
    def first(self) -> str:
        return "hello"

    @listen(first)
    def second(self, value: str) -> str:
        return value + " world"


# -- who is first-run collection for ------------------------------------------


def test_a_server_never_buffers_a_first_run(monkeypatch, grants) -> None:
    """No terminal, no host screen: nobody can answer, so the run is not
    buffered at all, uploads nothing, and the machine stays not asked."""
    buffer = Mock(side_effect=AssertionError("buffered a run nobody can approve"))
    monkeypatch.setattr(ephemeral, "EphemeralSpanBuffer", buffer)

    assert utils.should_auto_collect_first_time_traces() is False
    assert ExampleFlow().kickoff() == "hello world"
    assert ExampleFlow().kickoff() == "hello world"

    buffer.assert_not_called()
    assert grants == []
    assert user_data._load_user_data() == {}
    assert utils.is_first_execution() is True
    assert user_data.has_user_declined_tracing() is False


@pytest.mark.usefixtures("person_at_terminal")
def test_after_unasked_runs_the_first_terminal_run_asks(
    monkeypatch, grants, terminal_answer
) -> None:
    monkeypatch.setattr(utils, "_is_interactive_terminal", lambda: False)
    assert ExampleFlow().kickoff() == "hello world"
    terminal_answer.assert_not_called()

    monkeypatch.setattr(utils, "_is_interactive_terminal", lambda: True)
    assert utils.should_auto_collect_first_time_traces() is True
    assert ExampleFlow().kickoff() == "hello world"

    terminal_answer.assert_called_once_with(sharing=True)
    assert grants == []
    assert user_data._load_user_data() == ASKED_NO


@pytest.mark.parametrize(
    ("recorded", "collects"),
    [
        (None, True),
        (LEGACY_NO, True),
        (LEGACY_YES, False),
        (ASKED_NO, False),
        (ASKED_YES, False),
    ],
)
@pytest.mark.usefixtures("person_at_terminal")
def test_a_terminal_run_collects_only_until_someone_answered(
    recorded, collects
) -> None:
    if recorded is not None:
        _record(recorded)
    assert utils.should_auto_collect_first_time_traces() is collects


@pytest.mark.parametrize("recorded", [None, LEGACY_NO])
def test_a_host_consent_screen_counts_as_someone_to_ask(recorded) -> None:
    """The terminal UI silences crewAI's prompt and asks on its own screen."""
    if recorded is not None:
        _record(recorded)
    assert utils.should_auto_collect_first_time_traces() is False
    with trace_consent(lambda: False):
        assert utils.should_auto_collect_first_time_traces() is True


@pytest.mark.usefixtures("person_at_terminal")
def test_suppressed_messages_are_nobody_to_ask() -> None:
    token = utils.set_suppress_tracing_messages(True)
    try:
        assert utils.should_auto_collect_first_time_traces() is False
    finally:
        utils._suppress_tracing_messages.reset(token)


# -- what a first run records ---------------------------------------------------


def test_an_unasked_first_run_records_nothing_and_uploads_nothing(grants) -> None:
    _first_run_traced()

    assert grants == []
    assert user_data._load_user_data() == {}
    assert utils.is_first_execution() is True


def test_a_legacy_record_survives_an_unasked_run_unchanged(grants) -> None:
    _record(LEGACY_NO)
    _first_run_traced()

    assert grants == []
    assert user_data._load_user_data() == LEGACY_NO
    assert user_data.has_user_declined_tracing() is False


@pytest.mark.parametrize("approved", [False, True])
@pytest.mark.parametrize("recorded", [None, LEGACY_NO])
@pytest.mark.usefixtures("person_at_terminal")
def test_an_answer_at_the_terminal_is_recorded_as_asked(
    grants, terminal_answer, approved, recorded
) -> None:
    if recorded is not None:
        _record(recorded)
    terminal_answer.return_value = approved

    _first_run_traced()

    terminal_answer.assert_called_once_with(sharing=True)
    assert len(grants) == int(approved)
    assert user_data._load_user_data() == (ASKED_YES if approved else ASKED_NO)
    assert user_data.has_user_declined_tracing() is (not approved)
    assert utils.is_first_execution() is False


@pytest.mark.usefixtures("person_at_terminal")
def test_a_question_left_to_time_out_is_not_now(monkeypatch, grants) -> None:
    """The question was shown and nobody typed: uploads nothing and, as the
    "(20s timeout)" prompt and the docs say, is saved as Not now."""
    typing = threading.Event()
    monkeypatch.setattr("builtins.input", lambda: "y" if typing.wait(10) else "")
    monkeypatch.setattr(
        ephemeral,
        "prompt_user_for_trace_viewing",
        partial(utils.prompt_user_for_trace_viewing, 1),
    )

    try:
        _first_run_traced()
    finally:
        typing.set()

    assert grants == []
    assert user_data._load_user_data() == ASKED_NO
    assert user_data.has_user_declined_tracing() is True
    assert utils.is_first_execution() is False


@pytest.mark.usefixtures("person_at_terminal")
def test_a_no_at_the_terminal_is_final(monkeypatch, grants, terminal_answer) -> None:
    _record(ASKED_NO)

    assert ExampleFlow().kickoff() == "hello world"

    terminal_answer.assert_not_called()
    assert grants == []
    assert utils.should_auto_collect_first_time_traces() is False
    assert user_data._load_user_data() == ASKED_NO


@pytest.mark.parametrize(("answer", "recorded"), [(False, ASKED_NO), (True, ASKED_YES)])
def test_a_host_answer_is_recorded_as_asked(grants, answer, recorded) -> None:
    with trace_consent(lambda: answer):
        _first_run_traced()

    assert len(grants) == int(answer)
    assert user_data._load_user_data() == recorded


def test_a_host_that_closed_before_asking_records_nothing(grants) -> None:
    """The terminal UI returns None when the person quit the app: quitting is
    not an answer."""
    with trace_consent(lambda: None):
        _first_run_traced()

    assert grants == []
    assert user_data._load_user_data() == {}


@pytest.mark.usefixtures("person_at_terminal")
def test_turning_tracing_on_is_still_the_yes(monkeypatch, grants, terminal_answer):
    """`CREWAI_TRACING_ENABLED` is unchanged: no question, the run uploads."""
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "true")
    assert ExampleFlow().kickoff() == "hello world"

    terminal_answer.assert_not_called()
    assert len(grants) == 1


# -- the legacy event-bus first-run handler --------------------------------------


def _handler() -> FirstTimeTraceHandler:
    handler = FirstTimeTraceHandler()
    handler.is_first_time = True
    handler.collected_events = True
    return handler


def test_the_legacy_handler_neither_asks_nor_records_without_a_terminal(
    monkeypatch,
) -> None:
    prompt = Mock(side_effect=AssertionError("asked with nobody there"))
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.first_time_trace_handler.prompt_user_for_trace_viewing",
        prompt,
    )

    _handler().handle_execution_completion()

    prompt.assert_not_called()
    assert user_data._load_user_data() == {}


@pytest.mark.parametrize("recorded", [None, LEGACY_NO])
@pytest.mark.usefixtures("person_at_terminal")
def test_the_legacy_handler_records_a_terminal_no_as_asked(
    monkeypatch, recorded
) -> None:
    if recorded is not None:
        _record(recorded)
    monkeypatch.setattr(
        "crewai.events.listeners.tracing.first_time_trace_handler.prompt_user_for_trace_viewing",
        lambda: False,
    )
    monkeypatch.setattr(FirstTimeTraceHandler, "_show_tracing_declined_message", Mock())

    _handler().handle_execution_completion()

    data = user_data._load_user_data()
    assert data["trace_consent"] is False
    assert data["trace_consent_asked"] is True
    assert user_data.has_user_declined_tracing() is True


@pytest.mark.parametrize("recorded", [None, LEGACY_NO])
@pytest.mark.usefixtures("person_at_terminal")
def test_the_legacy_handler_records_nothing_when_the_question_fails(
    monkeypatch, recorded
) -> None:
    if recorded is not None:
        _record(recorded)
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError()))
    declined = Mock()
    monkeypatch.setattr(FirstTimeTraceHandler, "_show_tracing_declined_message", declined)

    _handler().handle_execution_completion()

    declined.assert_not_called()
    assert user_data._load_user_data() == (recorded or {})
    assert user_data.has_user_declined_tracing() is False
    assert utils.is_first_execution() is True


def test_mark_first_execution_done_keeps_an_earlier_answer() -> None:
    _record(ASKED_YES)
    utils.mark_first_execution_done(user_consented=False)
    assert user_data._load_user_data() == ASKED_YES


def test_saving_a_trace_answer_requires_the_answer_by_keyword() -> None:
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        utils._record_trace_answer()  # type: ignore[call-arg]  # the call under test
    with pytest.raises(TypeError, match="takes 0 positional arguments"):
        utils._record_trace_answer(False)  # type: ignore[misc]  # the call under test
    assert user_data._load_user_data() == {}


@pytest.mark.parametrize(
    "mark", [utils.mark_first_execution_done, utils.mark_first_execution_completed]
)
@pytest.mark.parametrize("enable", [False, True])
def test_the_public_mark_functions_keep_their_positional_signature(
    mark: Callable[[bool], None], enable: bool
) -> None:
    mark(enable)
    data = user_data._load_user_data()
    assert (data["trace_consent"], data["trace_consent_asked"]) == (enable, True)


@pytest.mark.parametrize("enable", [False, True])
@pytest.mark.usefixtures("person_at_terminal")
def test_the_public_first_run_confirmation_saves_the_answer_given(
    monkeypatch, enable
) -> None:
    monkeypatch.setattr(utils.click, "confirm", Mock(return_value=enable))

    assert utils.on_first_execution_tracing_confirmation() is enable

    data = user_data._load_user_data()
    assert (data["trace_consent"], data["trace_consent_asked"]) == (enable, True)
    assert user_data.has_user_declined_tracing() is (not enable)
    assert utils.on_first_execution_tracing_confirmation() is False


def test_the_public_first_run_confirmation_needs_someone_to_ask(monkeypatch) -> None:
    monkeypatch.setattr(
        utils.click,
        "confirm",
        Mock(side_effect=AssertionError("asked with nobody there")),
    )

    assert utils.on_first_execution_tracing_confirmation() is False
    assert user_data._load_user_data() == {}
    assert utils.is_first_execution() is True


# -- the question itself ----------------------------------------------------------


@pytest.mark.parametrize("sharing", [False, True])
@pytest.mark.parametrize(
    ("typed", "approved"), [("y", True), ("yes", True), ("n", False), ("", False)]
)
@pytest.mark.usefixtures("person_at_terminal")
def test_the_terminal_question_says_what_the_upload_gives(
    monkeypatch, capsys, sharing, typed, approved
) -> None:
    monkeypatch.setenv("COLUMNS", "240")
    monkeypatch.setattr("builtins.input", lambda: typed)

    assert utils.prompt_user_for_trace_viewing(sharing=sharing) is approved

    out = " ".join(capsys.readouterr().out.split())
    assert "See this run step by step?" in out
    assert (
        "Use CrewAI AMP to observe every agent decision, tool call and LLM call."
        in out
    )
    assert "Anyone with the link" not in out
    assert "24 hours" not in out
    assert "View trace? [y/N] (20s timeout):" in out


@pytest.mark.usefixtures("person_at_terminal")
def test_the_link_notice_holds_with_a_saved_login(monkeypatch, grants) -> None:
    """A first run uploads anonymously even when the CLI is logged in, so the
    24-hour, anyone-with-the-link notice is true for it too."""
    monkeypatch.setattr(
        "crewai.telemetry.tracing.grants.get_auth_token", lambda: "saved-login"
    )
    tiers: list[str] = []
    real_init = TraceGrantClient.__init__

    def init(self, amp_credential, **kwargs):
        real_init(self, amp_credential, **kwargs)
        tiers.append(self._tier)

    monkeypatch.setattr(TraceGrantClient, "__init__", init)
    monkeypatch.setattr(ephemeral, "prompt_user_for_trace_viewing", lambda **_: True)

    assert ExampleFlow().kickoff() == "hello world"

    assert tiers == ["ephemeral"] and len(grants) == 1


def test_the_question_is_not_shown_where_nobody_can_answer(monkeypatch) -> None:
    monkeypatch.setattr(
        "builtins.input", Mock(side_effect=AssertionError("read input with no one there"))
    )
    assert utils.prompt_user_for_trace_viewing() is None


def _console_that_cannot_render() -> Mock:
    console = Mock()
    console.return_value.print.side_effect = RuntimeError("terminal cannot render")
    return console


@pytest.mark.parametrize(
    "error", [EOFError(), OSError("stdin closed"), KeyboardInterrupt(), LookupError()]
)
@pytest.mark.usefixtures("person_at_terminal")
def test_a_question_whose_answer_cannot_be_read_is_not_answered(
    monkeypatch, error
) -> None:
    """Reading the answer failed: that is no answer, not a no."""
    monkeypatch.setattr("builtins.input", Mock(side_effect=error))
    assert utils.prompt_user_for_trace_viewing(timeout_seconds=5) is None


@pytest.mark.usefixtures("person_at_terminal")
def test_a_question_that_cannot_be_rendered_is_not_answered(monkeypatch) -> None:
    monkeypatch.setattr(utils, "Console", _console_that_cannot_render())
    monkeypatch.setattr(
        "builtins.input", Mock(side_effect=AssertionError("read input after a failure"))
    )
    assert utils.prompt_user_for_trace_viewing() is None


@pytest.mark.parametrize("failure", ["render", "input"])
@pytest.mark.parametrize("recorded", [None, LEGACY_NO])
@pytest.mark.usefixtures("person_at_terminal")
def test_a_failed_question_records_nothing_and_asks_again(
    monkeypatch, grants, failure, recorded
) -> None:
    """The first-run upload path: a prompt that failed saves no decline, so
    the next run at a terminal asks again."""
    if recorded is not None:
        _record(recorded)
    if failure == "render":
        monkeypatch.setattr(utils, "Console", _console_that_cannot_render())
    else:
        monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError()))

    _first_run_traced()

    assert grants == []
    assert user_data._load_user_data() == (recorded or {})
    assert user_data.has_user_declined_tracing() is False
    assert utils.is_first_execution() is True
    assert utils.should_auto_collect_first_time_traces() is True


# -- callers of has_user_declined_tracing ----------------------------------------


@pytest.mark.parametrize("recorded", [None, LEGACY_NO, ASKED_NO])
def test_disabled_tracing_notices_read_the_same_in_every_unconsented_state(
    monkeypatch, capsys, recorded
) -> None:
    """Crew, Flow and the console formatter tell a run without tracing how to
    turn it on, whether or not anybody declined."""
    from crewai.crew import Crew
    from crewai.events.listeners.tracing.trace_listener import (
        TraceCollectionListener,
    )
    from crewai.events.utils.console_formatter import ConsoleFormatter

    if recorded is not None:
        _record(recorded)
    monkeypatch.setattr(TraceCollectionListener, "_instance", None)

    Crew._show_tracing_disabled_message()
    Flow._show_tracing_disabled_message()
    ConsoleFormatter()._show_tracing_disabled_message_if_needed()

    out = capsys.readouterr().out
    assert out.count("Tracing is disabled.") == 3
    assert out.count("crewai traces enable") == 3


# -- the one-time hint for a coding agent -----------------------------------------

AGENT_HINT = "To debug runs step by step, run `crewai traces enable`"


@pytest.fixture
def coding_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run under Claude Code: no terminal for the question, output read back."""
    monkeypatch.setenv("CLAUDECODE", "1")


@pytest.mark.parametrize("recorded", [None, LEGACY_NO])
@pytest.mark.usefixtures("coding_agent")
def test_a_coding_agent_is_told_once_how_to_turn_traces_on(
    capsys, grants, recorded
) -> None:
    if recorded is not None:
        _record(recorded)

    assert ExampleFlow().kickoff() == "hello world"
    first = capsys.readouterr()
    assert ExampleFlow().kickoff() == "hello world"
    second = capsys.readouterr()

    assert first.err.count(AGENT_HINT) == 1
    assert AGENT_HINT in first.err.splitlines()
    assert AGENT_HINT not in first.out
    assert AGENT_HINT not in second.err + second.out
    assert user_data._load_user_data() == {
        **(recorded or {}),
        "trace_consent_agent_hint_shown": True,
    }
    assert grants == []


@pytest.mark.usefixtures("coding_agent")
def test_the_hint_is_no_answer_to_the_trace_question(monkeypatch, capsys) -> None:
    """Being told is not consenting: tracing stays off, and the next run at a
    terminal still asks."""
    assert ExampleFlow().kickoff() == "hello world"
    assert AGENT_HINT in capsys.readouterr().err

    assert utils.should_enable_tracing() is False
    assert user_data.has_user_declined_tracing() is False
    assert utils.is_first_execution() is True
    monkeypatch.setattr(utils, "_is_interactive_terminal", lambda: True)
    assert utils.should_auto_collect_first_time_traces() is True


@pytest.mark.parametrize("recorded", [ASKED_YES, LEGACY_YES, ASKED_NO])
@pytest.mark.usefixtures("coding_agent")
def test_a_coding_agent_is_not_told_once_the_question_is_answered(
    capsys, recorded
) -> None:
    _record(recorded)

    assert ExampleFlow().kickoff() == "hello world"

    assert AGENT_HINT not in capsys.readouterr().err
    assert user_data._load_user_data() == recorded


@pytest.mark.parametrize("value", ["true", "false", "1", "0"])
@pytest.mark.usefixtures("coding_agent")
def test_a_coding_agent_is_not_told_when_the_env_var_decides(
    monkeypatch, capsys, grants, value
) -> None:
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", value)

    assert ExampleFlow().kickoff() == "hello world"

    assert AGENT_HINT not in capsys.readouterr().err
    assert "trace_consent_agent_hint_shown" not in user_data._load_user_data()


@pytest.mark.usefixtures("coding_agent", "person_at_terminal")
def test_a_coding_agent_at_a_terminal_gets_the_question_not_the_hint(
    capsys, grants, terminal_answer
) -> None:
    assert ExampleFlow().kickoff() == "hello world"

    terminal_answer.assert_called_once_with(sharing=True)
    assert AGENT_HINT not in capsys.readouterr().err
    assert user_data._load_user_data() == ASKED_NO


def test_a_run_without_a_coding_agent_is_not_told(capsys) -> None:
    assert ExampleFlow().kickoff() == "hello world"

    assert AGENT_HINT not in capsys.readouterr().err
    assert user_data._load_user_data() == {}


@pytest.mark.usefixtures("coding_agent")
def test_a_host_that_silenced_tracing_messages_hides_the_hint(capsys) -> None:
    token = utils.set_suppress_tracing_messages(True)
    try:
        assert ExampleFlow().kickoff() == "hello world"
    finally:
        utils._suppress_tracing_messages.reset(token)

    assert AGENT_HINT not in capsys.readouterr().err
    assert user_data._load_user_data() == {}


@pytest.mark.usefixtures("coding_agent")
def test_the_hint_is_not_printed_when_it_cannot_be_recorded(
    monkeypatch, tmp_path, capsys
) -> None:
    """Unrecorded, it would print on every run; it waits for a run that can
    record it."""
    with monkeypatch.context() as unwritable:
        unwritable.setattr(
            utils, "_user_data_file", lambda: tmp_path / "missing-dir" / "user.json"
        )
        assert ExampleFlow().kickoff() == "hello world"
        assert AGENT_HINT not in capsys.readouterr().err

    assert ExampleFlow().kickoff() == "hello world"
    assert capsys.readouterr().err.count(AGENT_HINT) == 1


@pytest.mark.usefixtures("coding_agent")
def test_enabling_traces_after_the_hint_records_an_asked_yes(
    monkeypatch, tmp_path, capsys
) -> None:
    monkeypatch.chdir(tmp_path)
    assert ExampleFlow().kickoff() == "hello world"
    assert AGENT_HINT in capsys.readouterr().err

    result = CliRunner().invoke(traces_enable)

    assert result.exit_code == 0, result.output
    assert "Trace collection enabled." in result.output
    data = user_data._load_user_data()
    assert (data["trace_consent"], data["trace_consent_asked"]) == (True, True)
    assert utils.should_enable_tracing() is True
    assert utils.is_first_execution() is False


# -- a recorded yes uploads with nobody at the screen --------------------------

TRACE_URL = "https://app.crewai.com/crewai_plus/ephemeral_trace_batches/run?access_code=TRACE-x"
LINK_LINE = f"View traces: {TRACE_URL}"
_REAL_USER_DATA_DIR = appdirs.user_data_dir

needs_account_database = pytest.mark.skipif(
    os.name == "nt", reason="the account database is POSIX"
)


def _homes(
    monkeypatch: pytest.MonkeyPatch, *, environment: Path, account: Path
) -> None:
    """HOME (and XDG_DATA_HOME) as the environment says, and the home the
    account database gives this user — what a project's `.env` cannot set."""
    import pwd

    monkeypatch.setattr("crewai_core.paths.appdirs.user_data_dir", _REAL_USER_DATA_DIR)
    monkeypatch.setenv("HOME", str(environment))
    monkeypatch.setenv("XDG_DATA_HOME", str(environment / ".local" / "share"))
    real = pwd.getpwuid(os.getuid())
    entry = pwd.struct_passwd((*real[:5], str(account), real[6]))
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: entry)


@pytest.fixture
def own_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The user-data file lives in this account's own data folder."""
    _homes(monkeypatch, environment=tmp_path / "home", account=tmp_path / "home")
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def linked_grants(grants: list[TraceGrant], monkeypatch: pytest.MonkeyPatch):
    """Upload grants that carry AMP's 24-hour viewer link."""
    create = TraceGrantClient.create

    def with_link(client: TraceGrantClient, execution_uuid: str) -> TraceGrant:
        grant = replace(create(client, execution_uuid), trace_url=TRACE_URL)
        grants[-1] = grant
        return grant

    monkeypatch.setattr(TraceGrantClient, "create", with_link)
    return grants


@needs_account_database
@pytest.mark.usefixtures("own_store", "coding_agent")
def test_a_coding_agent_that_enabled_traces_uploads_and_sees_the_link(
    capsys, linked_grants, terminal_answer
) -> None:
    """The hint's advice works: `crewai traces enable`, then a run with no
    terminal and no login uploads anonymously and prints the link whole, on
    one line of stderr, where the agent reads it."""
    result = CliRunner().invoke(traces_enable)
    assert result.exit_code == 0, result.output
    capsys.readouterr()

    assert ExampleFlow().kickoff() == "hello world"

    terminal_answer.assert_not_called()
    assert len(linked_grants) == 1
    output = capsys.readouterr()
    assert LINK_LINE in output.err.splitlines()
    assert TRACE_URL not in output.out


@needs_account_database
@pytest.mark.parametrize("recorded", [LEGACY_YES, ASKED_NO])
@pytest.mark.usefixtures("own_store", "coding_agent")
def test_only_an_asked_yes_uploads_without_a_terminal(
    capsys, linked_grants, recorded
) -> None:
    """A yes saved before `trace_consent_asked` existed cannot be told apart
    from one nobody gave; a no is a no."""
    _record(recorded)

    assert ExampleFlow().kickoff() == "hello world"

    assert linked_grants == []
    assert "View traces:" not in capsys.readouterr().err
    assert user_data._load_user_data() == recorded


@needs_account_database
@pytest.mark.parametrize("value", ["false", "0"])
@pytest.mark.usefixtures("own_store", "coding_agent")
def test_tracing_turned_off_outranks_a_recorded_yes(
    monkeypatch, linked_grants, value
) -> None:
    _record(ASKED_YES)
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", value)

    assert ExampleFlow().kickoff() == "hello world"
    # And a run already buffering when the variable turned tracing off.
    with ephemeral_tracing(str(uuid4())) as session:
        session.get_tracer().start_span("execute flow").end()

    assert linked_grants == []


@needs_account_database
@pytest.mark.usefixtures("own_store", "coding_agent")
def test_a_host_that_says_no_outranks_a_recorded_yes(linked_grants) -> None:
    _record(ASKED_YES)

    with trace_consent(lambda: False):
        assert ExampleFlow().kickoff() == "hello world"

    assert linked_grants == []


@needs_account_database
@pytest.mark.parametrize("planted", ["home", "storage_dir"])
@pytest.mark.usefixtures("coding_agent")
def test_a_yes_in_a_store_the_environment_redirected_is_not_recorded(
    monkeypatch, tmp_path, linked_grants, planted
) -> None:
    """A project's `.env` can move the user-data file — HOME, XDG_DATA_HOME,
    or a CREWAI_STORAGE_DIR that leaves the data folder — to one holding a yes
    nobody on this account gave. Nobody can be asked, so nothing uploads."""
    monkeypatch.chdir(tmp_path)
    if planted == "home":
        _homes(monkeypatch, environment=tmp_path / "planted", account=tmp_path / "home")
    else:
        _homes(monkeypatch, environment=tmp_path / "home", account=tmp_path / "home")
        monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "project" / ".store"))
    _record(ASKED_YES)
    assert tmp_path / "home" not in user_data._user_data_file().parents

    assert ExampleFlow().kickoff() == "hello world"

    assert linked_grants == []


@needs_account_database
@pytest.mark.usefixtures("own_store", "person_at_terminal")
def test_a_recorded_yes_at_a_terminal_uploads_without_asking_again(
    capsys, linked_grants, terminal_answer
) -> None:
    _record(ASKED_YES)

    assert ExampleFlow().kickoff() == "hello world"

    terminal_answer.assert_not_called()
    assert len(linked_grants) == 1
    output = capsys.readouterr()
    assert "Ephemeral Execution Traces" in output.out
    assert "View traces:" not in output.err


@needs_account_database
@pytest.mark.usefixtures("own_store")
def test_tracing_asked_for_still_ignores_a_recorded_yes() -> None:
    """Only the env var and `tracing=True` are `tracing_asked_for`; the
    recorded yes is read where the upload is approved."""
    _record(ASKED_YES)

    assert utils.tracing_asked_for() is False
