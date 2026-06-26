"""Full-screen Textual TUI for crew execution.

Two-column layout: left sidebar (tasks/agents/tokens) + main content
(task header, plan checklist, activity timeline, streaming output).
"""

from collections.abc import Iterable
import json as _json
import re
import threading
import time
from typing import Any, ClassVar, cast

from crewai_core.telemetry import Telemetry
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Static


_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

# CrewAI brand palette
_C_PRIMARY = "#FF5A50"  # crewai.primary (coral red)
_C_TEAL = "#1F7982"  # crewai.secondary / tertiary
_C_GREEN = "#4aba6a"  # success green (warm, not neon)
_C_RED = "#FF5A50"  # error (same as primary)
_C_TEXT = "#e0e0e0"  # light text on dark bg
_C_DIM = "#AAAAAA"  # crewai.background-grey
_C_MUTED = "#666666"  # dimmer than _C_DIM for past timeline

_STEP_NUMBER_RE = re.compile(r"\bstep\s+(\d+)\b", re.IGNORECASE)
_REFINEMENT_RE = re.compile(r"^\s*step\s+(\d+)\s*:\s*(.+)\s*$", re.IGNORECASE)
_INTERNAL_TOOL_NAMES = {"create_reasoning_plan"}
_LOG_ARGS_TEXT_LIMIT = 3_000
_LOG_RESULT_TEXT_LIMIT = 5_000
_LOG_TRUNCATION_SUFFIX = "... [truncated]"
# Background memory saves can emit their start event just after kickoff returns.
_MEMORY_SAVE_DRAIN_GRACE_SECONDS = 2.0


def _is_save_to_memory_tool(tool_name: str | None) -> bool:
    return (tool_name or "").replace(" ", "_").lower() == "save_to_memory"


def _is_streaming_output(value: Any) -> bool:
    if not isinstance(value, Iterable):
        return False

    value_type = type(value)
    try:
        value_type.get_full_text  # noqa: B018
        value_type.result  # noqa: B018
    except AttributeError:
        return False
    return True


def _truncate_log_text(value: Any, limit: int) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) <= limit:
        return text
    suffix = _LOG_TRUNCATION_SUFFIX
    return f"{text[: max(0, limit - len(suffix))]}{suffix}"


def _enable_tracing_in_dotenv() -> None:
    """Append CREWAI_TRACING_ENABLED=true to .env if not already set."""
    from pathlib import Path

    env_file = Path.cwd() / ".env"
    key = "CREWAI_TRACING_ENABLED"
    try:
        if env_file.exists():
            content = env_file.read_text()
            if key in content:
                return
            sep = "" if content.endswith("\n") or not content else "\n"
            env_file.write_text(f"{content}{sep}{key}=true\n")
        else:
            env_file.write_text(f"{key}=true\n")
    except OSError:
        # Persisting the tracing flag is best-effort; an unwritable .env
        # must not block the run (tracing stays enabled for this session).
        pass


def _unescape_text(s: str) -> str:
    """Replace literal backslash-n sequences with real newlines."""
    return s.replace("\\n", "\n").replace("\\t", "  ")


def _try_parse_structured(text: str) -> Any | None:
    """Try JSON first, then Python repr (single-quoted dicts/lists)."""
    try:
        return _json.loads(text)
    except (ValueError, TypeError):
        pass
    try:
        import ast

        obj = ast.literal_eval(text)
        if isinstance(obj, (dict, list)):
            return obj
    except Exception:  # noqa: S110
        pass
    return None


def _format_json_in_text(text: str) -> str:
    """Find JSON objects/arrays in text and pretty-print them."""
    if not text or ("{" not in text and "[" not in text):
        return text

    result: list[str] = []
    i = 0
    while i < len(text):
        if text[i] in ("{", "["):
            close = "}" if text[i] == "{" else "]"
            depth = 0
            for j in range(i, len(text)):
                if text[j] == text[i]:
                    depth += 1
                elif text[j] == close:
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        parsed = _try_parse_structured(candidate)
                        if parsed is not None:
                            formatted = _json.dumps(
                                parsed, indent=2, ensure_ascii=False
                            )
                            result.append(formatted)
                            i = j + 1
                        else:
                            result.append(text[i])
                            i += 1
                        break
            else:
                remaining = text[i:]
                parsed = _try_parse_structured(remaining)
                if parsed is not None:
                    result.append(_json.dumps(parsed, indent=2, ensure_ascii=False))
                else:
                    result.append(remaining)
                break
        else:
            result.append(text[i])
            i += 1

    return "".join(result)


def _colorize_json_line(t: Text, line: str) -> None:
    """Append a single line with soft JSON syntax highlighting."""
    stripped = line.lstrip()
    leading = line[: len(line) - len(stripped)]
    t.append(leading, style=_C_MUTED)
    if not stripped:
        return
    s = stripped
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '"':
            j = i + 1
            while j < len(s):
                if s[j] == "\\":
                    j += 2
                    continue
                if s[j] == '"':
                    j += 1
                    break
                j += 1
            token = s[i:j]
            rest = s[j:].lstrip()
            if rest.startswith(":"):
                t.append(token, style=_C_TEAL)
            else:
                t.append(token, style=_C_DIM)
            i = j
        elif ch in "{}[],":
            t.append(ch, style=_C_MUTED)
            i += 1
        elif ch == ":":
            t.append(": ", style=_C_MUTED)
            i += 1
            if i < len(s) and s[i] == " ":
                i += 1
        elif ch in "-0123456789":
            j = i + 1
            while j < len(s) and s[j] in "0123456789.eE+-":
                j += 1
            t.append(s[i:j], style=_C_PRIMARY)
            i = j
        elif s[i : i + 4] == "true":
            t.append("true", style=_C_GREEN)
            i += 4
        elif s[i : i + 5] == "false":
            t.append("false", style=_C_GREEN)
            i += 5
        elif s[i : i + 4] == "null":
            t.append("null", style=f"italic {_C_MUTED}")
            i += 4
        else:
            t.append(ch, style=_C_DIM)
            i += 1


def _append_highlighted(t: Text, content: str, indent: str, max_lines: int = 50) -> int:
    """Append text with JSON highlighting if it looks like JSON, else plain."""
    lines = content.split("\n")
    total = len(lines)
    is_json = content.lstrip()[:1] in ("{", "[", '"')
    for line in lines[:max_lines]:
        t.append(f"{indent}  ", style="")
        if is_json:
            _colorize_json_line(t, line)
        else:
            t.append(line, style=_C_DIM)
        t.append("\n")
    return total


class TraceConsentScreen(ModalScreen[bool]):
    CSS = """
    TraceConsentScreen {
        align: center middle;
    }
    #consent-dialog {
        width: 50;
        height: auto;
        max-height: 16;
        background: #1c1c1c;
        border: tall #333333;
        padding: 1 2 2 2;
    }
    #consent-buttons {
        height: 3;
        margin-top: 1;
        width: 100%;
        layout: horizontal;
    }
    .consent-btn {
        width: 1fr;
        height: 3;
        margin: 0 1;
    }
    #btn-consent-yes {
        background: #1F7982;
        color: #e0e0e0;
        border: none;
        text-style: bold;
    }
    #btn-consent-yes:hover {
        background: #28969f;
    }
    #btn-consent-yes:disabled {
        background: #1F7982;
        color: #e0e0e0;
        text-opacity: 100%;
        opacity: 100%;
    }
    #btn-consent-no {
        background: #333333;
        color: #AAAAAA;
        border: none;
    }
    #btn-consent-no:hover {
        background: #444444;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("y", "consent_yes", "Yes", show=False),
        Binding("n", "consent_no", "No", show=False),
        Binding("escape", "consent_no", "Cancel", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._sending = False
        self._frame = 0
        self._spin_timer: Any = None

    def compose(self) -> ComposeResult:
        with Vertical(id="consent-dialog"):
            yield Static(self._build_content(), id="consent-text")
            with Horizontal(id="consent-buttons"):
                yield Button("View Traces", id="btn-consent-yes", classes="consent-btn")
                yield Button("Cancel", id="btn-consent-no", classes="consent-btn")

    def _build_content(self) -> Text:
        t = Text()
        t.append("  View execution traces on CrewAI AMP\n\n", style=f"bold {_C_TEXT}")
        t.append("  Sends agent decisions, tool calls, and\n", style=_C_DIM)
        t.append("  timing data. Link expires in 24h.\n\n", style=_C_DIM)
        t.append("  Traces will be enabled for future runs.\n", style=_C_MUTED)
        return t

    def _start_sending(self) -> None:
        self._sending = True
        btn_yes = self.query_one("#btn-consent-yes", Button)
        btn_no = self.query_one("#btn-consent-no", Button)
        btn_yes.disabled = True
        btn_yes.label = f"{_SPINNER[0]} Loading…"
        btn_no.display = False
        self._spin_timer = self.set_interval(1 / 8, self._spin_tick)
        cast("CrewRunApp", self.app)._on_trace_consent_accepted()

    def _spin_tick(self) -> None:
        self._frame += 1
        try:
            btn = self.query_one("#btn-consent-yes", Button)
            btn.label = f"{_SPINNER[self._frame % len(_SPINNER)]} Loading…"
        except Exception:  # noqa: S110
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._sending:
            return
        if event.button.id == "btn-consent-yes":
            self._start_sending()
        else:
            self.dismiss(False)

    def action_consent_yes(self) -> None:
        if self._sending:
            return
        self._start_sending()

    def action_consent_no(self) -> None:
        if self._sending:
            return
        self.dismiss(False)


class CrewRunApp(App[Any]):
    TITLE = "CrewAI"

    CSS = """
Screen {
    background: #131313;
}

#body {
    height: 1fr;
}

#sidebar {
    width: 34;
    background: #1c1c1c;
    border-right: vkey #333333;
    scrollbar-size-vertical: 1;
    scrollbar-color: #666666;
    scrollbar-color-hover: #FF5A50;
    scrollbar-background: #1c1c1c;
    overflow-y: auto;
    overflow-x: hidden;
}

#sidebar-content {
    width: 100%;
    height: auto;
    padding: 1 0;
}

#main-panel {
    width: 1fr;
}

#task-header {
    height: auto;
    max-height: 6;
    padding: 1 2;
    background: #1c1c1c;
    border-bottom: hkey #333333;
}

#scroll-area {
    height: 3fr;
    min-height: 6;
    scrollbar-size-vertical: 1;
    scrollbar-color: #666666;
    scrollbar-color-hover: #FF5A50;
    scrollbar-background: #131313;
}

#main-content {
    padding: 1 2;
    height: auto;
}

#conversation-input {
    display: none;
    height: 3;
    border-top: hkey #333333;
    background: #1c1c1c;
    color: #e0e0e0;
}

#conversation-input:focus {
    border-top: hkey #1F7982;
}

Header {
    background: #1c1c1c;
    color: #FF5A50;
}

Footer {
    background: #1c1c1c;
}

FooterKey {
    background: #1c1c1c;
    color: #AAAAAA;
}

FooterKey .footer-key--key {
    background: #262626;
    color: #FF5A50;
}

#log-panel {
    height: 2fr;
    min-height: 6;
    background: #1c1c1c;
    border-top: hkey #333333;
    scrollbar-size-vertical: 1;
    scrollbar-color: #666666;
    scrollbar-color-hover: #FF5A50;
    scrollbar-background: #1c1c1c;
}

#log-content {
    padding: 1 2;
    height: auto;
}

#sidebar-actions {
    display: none;
    height: auto;
    padding: 0 1;
    margin-top: 1;
    border-top: hkey #333333;
}

.action-btn {
    width: 100%;
    min-width: 20;
    height: 3;
    margin: 1 1 0 1;
    text-style: bold;
}

#btn-traces {
    background: #1F7982;
    color: #e0e0e0;
    border: none;
}
#btn-traces:hover {
    background: #28969f;
}
#btn-traces:disabled {
    background: #1a4a50;
    color: #888888;
}

#btn-deploy {
    background: #333333;
    color: #e0e0e0;
    border: none;
}
#btn-deploy:hover {
    background: #444444;
}

#btn-traces-done {
    background: #1a3a3a;
    color: #1F7982;
    border: none;
}
#btn-traces-done:hover {
    background: #1F7982;
    color: #e0e0e0;
}
"""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "Quit"),
        Binding("s", "toggle_sidebar", "Sidebar"),
        Binding("l", "toggle_logs", "Logs"),
        Binding("t", "view_traces", "Traces", show=False),
        Binding("d", "deploy_crew", "Deploy", show=False),
        Binding("down", "log_down", "Log ↓", show=False),
        Binding("up", "log_up", "Log ↑", show=False),
        Binding("enter", "log_toggle", "Expand", show=False),
    ]

    def __init__(
        self,
        crew_name: str = "Crew",
        total_tasks: int = 0,
        agent_names: list[str] | None = None,
        task_names: list[str] | None = None,
        conversational: bool = False,
    ):
        super().__init__()
        self.title = f"CrewAI — {crew_name}"
        self.sub_title = "0:00"
        self._crew_name = crew_name
        self._lock = threading.RLock()

        self._total_tasks = total_tasks
        self._current_task_idx = 0
        self._current_task_desc = ""
        self._current_agent = ""
        self._task_names = task_names or []
        self._agent_names = agent_names or []
        self._task_statuses: dict[int, str] = {
            i: "pending" for i in range(1, total_tasks + 1)
        }
        # Maps a task's identity to state captured when it started (sidebar
        # index, description, agent, start time) so completion/failure events
        # build their log entry from the right task even when tasks run
        # async/overlapping.
        self._task_state_by_key: dict[str, dict[str, Any]] = {}

        self._timeline: list[tuple[str, str, str]] = []
        self._current_step: tuple[str, str, str] | None = None

        self._input_tokens = 0
        self._output_tokens = 0
        self._live_out_tokens = 0
        self._pending_input_estimate = 0
        self._llm_calls = 0

        self._streaming_text = ""
        self._is_streaming = False
        self._current_llm_text = ""
        self._task_full_output = ""

        self._plan: dict[str, Any] | None = None
        self._plan_step_status: dict[int, str] = {}
        self._awaiting_replan = False

        self._status = "starting"
        self._start_time = time.time()
        self._task_start_time = time.time()
        self._final_output: str | None = None
        self._error: str | None = None
        self._frame = 0

        self._task_logs: list[dict[str, Any]] = []
        self._current_task_steps: list[dict[str, Any]] = []

        self._log_entries: list[dict[str, Any]] = []
        self._log_cursor: int = 0
        self._log_expanded: set[int] = set()
        self._log_scroll_needed: bool = False
        self._log_line_map: list[tuple[int, int, int]] = []
        self._suppressed_memory_save_event_ids: set[str] = set()
        self._memory_save_drain_timer: Any = None

        self._event_handlers: list[tuple[type, Any]] = []

        self._crew: Any = None
        self._flow: Any = None
        self._is_conversational = conversational
        self._conversation_messages: list[tuple[str, str]] = []
        self._conversation_turns = 0
        self._conversation_turn_in_progress = False
        self._conversation_previous_defer_trace_finalization: bool | None = None
        self._conversation_exit_commands = {"exit", "quit"}
        self._default_inputs: dict[str, Any] | None = None
        self._crew_result: Any = None
        self._crew_json_path: Any = None
        self._elapsed_frozen: float | None = None
        self._want_deploy: bool = False
        self._trace_url: str | None = None
        self._consent_screen: TraceConsentScreen | None = None
        self._telemetry: Telemetry | None = None

    # ── Layout ──────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="body"):
            with VerticalScroll(id="sidebar"):
                yield Static(id="sidebar-content")
                with Vertical(id="sidebar-actions"):
                    yield Button("View Traces", id="btn-traces", classes="action-btn")
                    yield Button("Deploy", id="btn-deploy", classes="action-btn")
            with Vertical(id="main-panel"):
                yield Static(id="task-header")
                with VerticalScroll(id="scroll-area"):
                    yield Static(id="main-content")
                yield Input(
                    placeholder="Message the flow...",
                    id="conversation-input",
                )
                with VerticalScroll(id="log-panel"):
                    yield Static(id="log-content")
        yield Footer()

    def on_mount(self) -> None:
        self._start_time = time.time()
        self._subscribe()
        self._tick_timer = self.set_interval(1 / 8, self._tick)
        if self._is_conversational and self._flow:
            self._start_conversational_session()
        elif self._crew:
            self._run_crew_worker()
        elif self._crew_json_path:
            self._load_and_run_worker()

    # ── Crew execution ──────────────────────────────────────

    @work(thread=True, exclusive=True, group="crew")
    def _load_and_run_worker(self) -> None:
        from crewai.events.listeners.tracing.utils import (
            set_suppress_tracing_messages,
            set_tui_mode,
        )

        set_tui_mode(True)
        set_suppress_tracing_messages(True)
        try:
            from crewai.project.crew_loader import load_crew

            crew, default_inputs = load_crew(self._crew_json_path)
            crew.verbose = False
            for agent in crew.agents:
                agent.verbose = False
                if hasattr(agent, "llm") and hasattr(agent.llm, "stream"):
                    agent.llm.stream = True

            task_names = []
            for task in crew.tasks:
                name = getattr(task, "name", "") or ""
                if not name:
                    desc = getattr(task, "description", "") or "Task"
                    name = desc[:40]
                task_names.append(name)

            agent_names = []
            for agent in crew.agents:
                name = (
                    getattr(agent, "role", "") or getattr(agent, "name", "") or "Agent"
                )
                agent_names.append(name)

            self._crew = crew
            self._default_inputs = default_inputs

            def _apply_crew_info() -> None:
                with self._lock:
                    self._total_tasks = len(crew.tasks)
                    self._task_names = task_names
                    self._agent_names = agent_names
                    self._task_statuses = {
                        i: "pending" for i in range(1, len(crew.tasks) + 1)
                    }
                    self.title = f"CrewAI — {crew.name or 'Crew'}"
                    self._crew_name = crew.name or "Crew"
                self._start_time = time.time()
                self._run_crew_worker()

            self.call_from_thread(_apply_crew_info)
        except Exception as e:
            self.call_from_thread(self._on_crew_failed, str(e))

    @work(thread=True, exclusive=True, group="crew")
    def _run_crew_worker(self) -> None:
        from crewai.events.listeners.tracing.utils import (
            set_suppress_tracing_messages,
            set_tui_mode,
        )

        set_tui_mode(True)
        set_suppress_tracing_messages(True)
        try:
            result = self._crew.kickoff(inputs=self._default_inputs)
            output = result.raw if result and hasattr(result, "raw") else None
            with self._lock:
                self._crew_result = result
            self.call_from_thread(self._on_crew_done, output)
        except Exception as e:
            self.call_from_thread(self._on_crew_failed, str(e))

    def _on_crew_done(self, output: str | None) -> None:
        with self._lock:
            self._status = "completed"
            self._final_output = output
            self._is_streaming = False
            self._streaming_text = ""
            self._current_step = None
            self._timeline = []
            self._elapsed_frozen = time.time() - self._start_time
            self._collapse_plan_on_task_done()
            for k in self._task_statuses:
                if self._task_statuses[k] == "active":
                    self._task_statuses[k] = "done"
            now = time.time()
            for entry in self._log_entries:
                if entry["status"] == "running":
                    if entry["tool_name"] == "memory_save":
                        continue
                    entry["status"] = "timeout"
                    entry["error"] = "No result received before crew completed"
                    entry["duration"] = now - entry["start_time"]
        try:
            from crewai.events.listeners.tracing.trace_listener import (
                TraceCollectionListener,
            )

            listener: TraceCollectionListener | None = getattr(
                TraceCollectionListener, "_instance", None
            )
            if listener and listener.batch_manager:
                bm = listener.batch_manager
                self._trace_url = (
                    getattr(bm, "trace_url", None) or bm.ephemeral_trace_url
                )
        except Exception:  # noqa: S110
            pass
        try:
            self.query_one("#sidebar-actions").display = True
            if self._trace_url:
                btn = self.query_one("#btn-traces", Button)
                btn.label = "✔ Open Traces"
                btn.id = "btn-traces-done"
        except Exception:  # noqa: S110
            pass
        self._tick()
        self._scroll_to_result()
        self.call_later(self._focus_activity_log)
        self._tick_timer.stop()
        self._tick_timer = self.set_interval(1 / 2, self._tick)
        self._unsubscribe_if_no_running_memory_save(wait_for_queued=True)

    def _on_crew_failed(self, error: str) -> None:
        with self._lock:
            self._status = "failed"
            self._error = error
            self._is_streaming = False
            self._current_step = None
            self._elapsed_frozen = time.time() - self._start_time
            now = time.time()
            for entry in self._log_entries:
                if entry["status"] == "running":
                    if entry["tool_name"] == "memory_save":
                        continue
                    entry["status"] = "error"
                    entry["error"] = "No result received before crew failed"
                    entry["duration"] = now - entry["start_time"]
        self._tick()
        self.call_later(self._focus_activity_log)
        self._tick_timer.stop()
        self._tick_timer = self.set_interval(1 / 2, self._tick)
        self._unsubscribe_if_no_running_memory_save(wait_for_queued=True)

    # ── Conversational flow execution ───────────────────────

    def _start_conversational_session(self) -> None:
        from crewai.events.listeners.tracing.utils import (
            set_suppress_tracing_messages,
            set_tui_mode,
        )

        set_tui_mode(True)
        set_suppress_tracing_messages(True)
        with self._lock:
            self._status = "chatting"
            self._current_step = None
            self._elapsed_frozen = None
            self._conversation_previous_defer_trace_finalization = getattr(
                self._flow, "defer_trace_finalization", False
            )
            self._flow.defer_trace_finalization = True

        try:
            input_widget = self.query_one("#conversation-input", Input)
            input_widget.display = True
            input_widget.focus()
        except Exception:  # noqa: S110
            pass

    def _finalize_conversational_session(self) -> None:
        if not (self._is_conversational and self._flow):
            return
        try:
            self._flow.finalize_session_traces()
        except Exception:  # noqa: S110
            pass
        previous = self._conversation_previous_defer_trace_finalization
        if previous is not None:
            try:
                self._flow.defer_trace_finalization = previous
            except Exception:  # noqa: S110
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id != "conversation-input":
            return
        if not self._is_conversational:
            return

        message = event.value.strip()
        event.input.value = ""
        if not message:
            return
        if message.lower() in self._conversation_exit_commands:
            self._finalize_conversational_session()
            self._unsubscribe()
            self.exit(self._crew_result)
            return
        if self._conversation_turn_in_progress:
            return

        with self._lock:
            self._conversation_messages.append(("user", message))
            self._conversation_turn_in_progress = True
            self._conversation_turns += 1
            self._status = "working"
            self._current_step = ("yellow", "Thinking…", "")
            self._is_streaming = False
            self._streaming_text = ""
            self._task_full_output = ""
            self._current_llm_text = ""

        event.input.disabled = True
        self._run_conversation_turn_worker(message)

    @work(thread=True, exclusive=True, group="conversation")
    def _run_conversation_turn_worker(self, message: str) -> None:
        from crewai.events.listeners.tracing.utils import (
            set_suppress_tracing_messages,
            set_tui_mode,
        )

        set_tui_mode(True)
        set_suppress_tracing_messages(True)
        try:
            result = self._flow.handle_turn(message)
            result = self._consume_conversation_streaming_result(result)
            self.call_from_thread(self._on_conversation_turn_done, result)
        except Exception as e:
            self.call_from_thread(self._on_conversation_turn_failed, str(e))

    def _consume_conversation_streaming_result(self, result: Any) -> Any:
        if not _is_streaming_output(result):
            return result
        for _chunk in result:
            pass
        return result.result

    def _on_conversation_turn_done(self, result: Any) -> None:
        with self._lock:
            output = self._stringify_output(result)
            self._conversation_messages.append(("assistant", output))
            self._crew_result = result
            self._conversation_turn_in_progress = False
            self._status = "chatting"
            self._is_streaming = False
            self._streaming_text = ""
            self._current_step = None
        self._enable_conversation_input()
        self._tick()
        self._scroll_to_result()

    def _on_conversation_turn_failed(self, error: str) -> None:
        with self._lock:
            self._status = "failed"
            self._error = error
            self._conversation_turn_in_progress = False
            self._is_streaming = False
            self._current_step = None
        self._enable_conversation_input()
        self._tick()

    def _enable_conversation_input(self) -> None:
        try:
            input_widget = self.query_one("#conversation-input", Input)
            input_widget.disabled = False
            input_widget.focus()
        except Exception:  # noqa: S110
            pass

    def _stringify_output(self, result: Any) -> str:
        raw_result = getattr(result, "raw", result)
        if raw_result is None:
            return ""
        if isinstance(raw_result, str):
            return raw_result
        try:
            return _json.dumps(raw_result, default=str, ensure_ascii=False)
        except TypeError:
            return str(raw_result)

    # ── Actions ─────────────────────────────────────────────

    def action_toggle_sidebar(self) -> None:
        sidebar = self.query_one("#sidebar")
        sidebar.display = not sidebar.display

    def action_toggle_logs(self) -> None:
        panel = self.query_one("#log-panel")
        panel.display = not panel.display

    def action_log_down(self) -> None:
        try:
            if not self.query_one("#log-panel").display:
                return
        except Exception:
            return
        should_refresh = False
        with self._lock:
            if self._log_entries:
                self._log_cursor = min(self._log_cursor + 1, len(self._log_entries) - 1)
                self._log_scroll_needed = True
                should_refresh = True
        if should_refresh:
            self._refresh_log_panel()

    def action_log_up(self) -> None:
        try:
            if not self.query_one("#log-panel").display:
                return
        except Exception:
            return
        should_refresh = False
        with self._lock:
            if self._log_entries:
                self._log_cursor = max(self._log_cursor - 1, 0)
                self._log_scroll_needed = True
                should_refresh = True
        if should_refresh:
            self._refresh_log_panel()

    def action_log_toggle(self) -> None:
        try:
            if not self.query_one("#log-panel").display:
                return
        except Exception:
            return
        should_refresh = False
        with self._lock:
            if self._log_entries:
                if self._log_cursor in self._log_expanded:
                    self._log_expanded.discard(self._log_cursor)
                else:
                    self._log_expanded.add(self._log_cursor)
                should_refresh = True
        if should_refresh:
            self._refresh_log_panel()

    async def action_quit(self) -> None:
        self._finalize_conversational_session()
        self._unsubscribe()
        self.exit(self._crew_result)

    def action_view_traces(self) -> None:
        if self._status != "completed":
            return
        if self._trace_url:
            import webbrowser

            try:
                webbrowser.open(self._trace_url)
            except Exception:  # noqa: S110
                pass
            return
        self._consent_screen = TraceConsentScreen()
        self.push_screen(self._consent_screen)

    def _on_trace_consent_accepted(self) -> None:
        self._send_traces_worker()

    @work(thread=True)
    def _send_traces_worker(self) -> None:
        import webbrowser

        try:
            from crewai.events.listeners.tracing.utils import (
                set_suppress_tracing_messages,
                set_tui_mode,
            )

            set_tui_mode(True)
            set_suppress_tracing_messages(True)

            from crewai.events.listeners.tracing.trace_listener import (
                TraceCollectionListener,
            )
            from crewai.events.listeners.tracing.utils import (
                mark_first_execution_completed,
            )

            listener: TraceCollectionListener | None = getattr(
                TraceCollectionListener, "_instance", None
            )
            if not listener:
                self.call_from_thread(self._dismiss_consent_modal)
                return

            bm = listener.batch_manager
            url = getattr(bm, "trace_url", None) or bm.ephemeral_trace_url

            if not url:
                handler = listener.first_time_handler
                handler.set_batch_manager(bm)
                handler._initialize_backend_and_send_events()
                url = handler.ephemeral_url or bm.ephemeral_trace_url

                if listener.first_time_handler.is_first_time:
                    mark_first_execution_completed(user_consented=True)

            _enable_tracing_in_dotenv()

            if url:
                self._trace_url = url

                def _done() -> None:
                    self._dismiss_consent_modal()
                    try:
                        btn = self.query_one("#btn-traces", Button)
                        btn.label = "✔ Open Traces"
                        btn.id = "btn-traces-done"
                    except Exception:  # noqa: S110
                        pass

                self.call_from_thread(_done)
                try:
                    webbrowser.open(url)
                except Exception:  # noqa: S110
                    pass
            else:
                self.call_from_thread(self._dismiss_consent_modal)
        except Exception:
            self.call_from_thread(self._dismiss_consent_modal)

    def _dismiss_consent_modal(self) -> None:
        try:
            screen = self._consent_screen
            if screen and screen.is_attached:
                screen.dismiss(False)
        except Exception:  # noqa: S110
            pass

    def action_deploy_crew(self) -> None:
        if self._status != "completed":
            return
        self._want_deploy = True
        self._unsubscribe()
        self.exit(self._crew_result)

    def _record_tui_button_click(self, button_name: str) -> None:
        try:
            if self._telemetry is None:
                self._telemetry = Telemetry()
                self._telemetry.set_tracer()
            self._telemetry.feature_usage_span(f"cli_usage:{button_name}")
        except Exception:  # noqa: S110
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id in ("btn-traces", "btn-traces-done"):
            self._record_tui_button_click("view_traces")
            self.action_view_traces()
        elif event.button.id == "btn-deploy":
            self._record_tui_button_click("deploy")
            self.action_deploy_crew()

    def _scroll_to_result(self) -> None:
        try:
            scroll = self.query_one("#scroll-area", VerticalScroll)
            self.call_later(lambda: scroll.scroll_end(animate=False))
        except Exception:  # noqa: S110
            pass

    def _focus_activity_log(self) -> None:
        if not self._is_mounted:
            return
        log_panel = self.query_one("#log-panel", VerticalScroll)
        if log_panel.display:
            log_panel.focus()

    def _refresh_log_panel(self) -> None:
        if not self._is_mounted:
            return
        with self._lock:
            if self.query_one("#log-panel").display:
                self._render_log_panel()

    def on_click(self, event: Any) -> None:
        try:
            widget = self.query_one("#log-content", Static)
        except Exception:
            return
        if not widget.region.contains(event.screen_x, event.screen_y):
            return
        scroll = self.query_one("#log-panel", VerticalScroll)
        clicked_line = event.screen_y - widget.region.y + int(scroll.scroll_y)
        with self._lock:
            for idx, start, end in self._log_line_map:
                if start <= clicked_line < end:
                    self._log_cursor = idx
                    if idx in self._log_expanded:
                        self._log_expanded.discard(idx)
                    else:
                        self._log_expanded.add(idx)
                    break
        self._refresh_log_panel()

    # ── Tick (8 fps) ────────────────────────────────────────

    def _tick(self) -> None:
        self._frame += 1
        elapsed = getattr(self, "_elapsed_frozen", None) or (
            time.time() - self._start_time
        )
        mins, secs = divmod(int(elapsed), 60)
        self.sub_title = f"{mins}:{secs:02d}"

        try:
            with self._lock:
                self._render_sidebar()
                self._render_task_header()
                self._render_main_content()
                if self.query_one("#log-panel").display:
                    self._render_log_panel()
        except NoMatches:
            return

    def _spinner(self) -> str:
        return _SPINNER[self._frame % len(_SPINNER)]

    # ── Sidebar rendering ───────────────────────────────────

    def _render_sidebar(self) -> None:
        widget = self.query_one("#sidebar-content", Static)
        t = Text()
        sidebar_width = 30

        if self._is_conversational:
            t.append("  CONVERSATION\n", style=f"bold {_C_PRIMARY}")
            t.append("\n")
            if self._conversation_turn_in_progress:
                t.append(f"  {self._spinner()} ", style=_C_PRIMARY)
                t.append("Working\n", style=f"bold {_C_TEXT}")
            elif self._status == "failed":
                t.append("  ✘ Failed\n", style=_C_RED)
            else:
                t.append("  ● Ready\n", style=_C_GREEN)
            t.append(f"  Turns {self._conversation_turns}\n", style=_C_DIM)
            t.append("\n")
            t.append("  TOKENS\n", style=f"bold {_C_PRIMARY}")
            t.append("\n")
            out = self._output_tokens + self._live_out_tokens
            t.append(f"  ↑ {self._input_tokens:,}\n", style=_C_DIM)
            t.append(f"  ↓ {out:,}\n", style=_C_DIM)
            t.append("\n")
            t.append("  COMMANDS\n", style=f"bold {_C_PRIMARY}")
            t.append("\n")
            t.append("  quit / exit\n", style=_C_DIM)
            widget.update(t)
            return

        t.append("  TASKS\n", style=f"bold {_C_PRIMARY}")
        t.append("\n")

        for i in range(1, self._total_tasks + 1):
            status = self._task_statuses.get(i, "pending")
            name = (
                self._task_names[i - 1] if i <= len(self._task_names) else f"Task {i}"
            )
            max_name = sidebar_width - 6
            if len(name) > max_name:
                name = name[: max_name - 1] + "…"

            if status == "done":
                t.append("  ✔ ", style=_C_GREEN)
                t.append(f"{name}\n", style=_C_DIM)
            elif status == "active":
                t.append(f"  {self._spinner()} ", style=_C_PRIMARY)
                t.append(f"{name}\n", style=f"bold {_C_TEXT}")
            elif status == "failed":
                t.append("  ✘ ", style=_C_RED)
                t.append(f"{name}\n", style=_C_RED)
            else:
                t.append("  ○ ", style=_C_DIM)
                t.append(f"{name}\n", style=_C_DIM)

        t.append("\n")
        t.append("  AGENTS\n", style=f"bold {_C_PRIMARY}")
        t.append("\n")

        for name in self._agent_names:
            max_name = sidebar_width - 6
            disp = name[: max_name - 1] + "…" if len(name) > max_name else name
            if name == self._current_agent:
                t.append(f"  ● {disp}\n", style=f"bold {_C_PRIMARY}")
            else:
                t.append(f"    {disp}\n", style=_C_DIM)

        t.append("\n")
        t.append("  TOKENS\n", style=f"bold {_C_PRIMARY}")
        t.append("\n")

        out = self._output_tokens + self._live_out_tokens
        t.append(f"  ↑ {self._input_tokens:,}\n", style=_C_DIM)
        t.append(f"  ↓ {out:,}\n", style=_C_DIM)

        widget.update(t)

    # ── Task header rendering ───────────────────────────────

    def _render_task_header(self) -> None:
        widget = self.query_one("#task-header", Static)
        t = Text()

        if self._is_conversational:
            if self._status == "failed":
                t.append("✘ ", style=f"bold {_C_RED}")
                t.append("Failed", style=f"bold {_C_RED}")
                if self._error:
                    t.append(f"\n{self._error[:120]}", style=_C_RED)
            elif self._conversation_turn_in_progress:
                t.append(f"{self._spinner()} ", style=_C_PRIMARY)
                t.append("Flow is responding", style=f"bold {_C_PRIMARY}")
            else:
                t.append("● ", style=f"bold {_C_GREEN}")
                t.append("Conversational flow ready", style=f"bold {_C_GREEN}")
                t.append("  Type a message below", style=_C_DIM)
            widget.update(t)
            return

        if self._status == "completed":
            elapsed = self._elapsed_frozen or (time.time() - self._start_time)
            t.append("✔ ", style=f"bold {_C_GREEN}")
            t.append(f"Completed {self._total_tasks} tasks", style=f"bold {_C_GREEN}")
            t.append(f"  {elapsed:.1f}s", style=_C_DIM)

            out = self._output_tokens + self._live_out_tokens
            parts = []
            if self._input_tokens:
                parts.append(f"↑{self._input_tokens:,}")
            if out:
                parts.append(f"↓{out:,}")
            if parts:
                t.append(f"  {' '.join(parts)} tokens", style=_C_DIM)

        elif self._status == "failed":
            t.append("✘ ", style=f"bold {_C_RED}")
            t.append("Failed", style=f"bold {_C_RED}")
            if self._error:
                t.append(f"\n{self._error[:120]}", style=_C_RED)

        elif self._current_task_idx > 0:
            t.append(
                f"Task {self._current_task_idx}/{self._total_tasks}",
                style=f"bold {_C_PRIMARY}",
            )
            if self._current_task_desc:
                desc = self._current_task_desc
                if len(desc) > 80:
                    desc = desc[:79] + "…"
                t.append(f"  —  {desc}", style=_C_TEXT)
            if self._current_agent:
                t.append("\nAgent: ", style=_C_DIM)
                t.append(self._current_agent, style=f"bold {_C_TEXT}")

        else:
            t.append(f"{self._spinner()} ", style=_C_PRIMARY)
            if not self._crew:
                t.append("Loading crew…", style=_C_DIM)
            else:
                t.append("Starting crew…", style=_C_DIM)

        widget.update(t)

    # ── Main content rendering ──────────────────────────────

    def _render_main_content(self) -> None:
        widget = self.query_one("#main-content", Static)
        t = Text()
        should_scroll = False

        if self._is_conversational:
            if not self._conversation_messages and not self._is_streaming:
                t.append("  Start the conversation below.\n", style=_C_MUTED)
            for role, content in self._conversation_messages:
                if role == "user":
                    t.append("\n  You\n", style=f"bold {_C_TEAL}")
                else:
                    t.append("\n  Assistant\n", style=f"bold {_C_PRIMARY}")
                rendered = _format_json_in_text(_unescape_text(content))
                for line in rendered.split("\n"):
                    style = _C_TEXT if role == "assistant" else _C_DIM
                    t.append(f"  {line}\n", style=style)

            if self._is_streaming and self._streaming_text:
                text = _unescape_text(self._filtered_streaming_text())
                if text.strip():
                    t.append("\n  Assistant\n", style=f"bold {_C_PRIMARY}")
                    for line in text.rstrip().split("\n")[-40:]:
                        t.append(f"  {line}\n", style=_C_TEXT)
                    should_scroll = True

            if self._status == "failed" and self._error:
                t.append("\n  Error\n", style=f"bold {_C_RED}")
                t.append(f"  {self._error}\n", style=_C_RED)

            widget.update(t)
            if should_scroll:
                try:
                    self.query_one("#scroll-area", VerticalScroll).scroll_end(
                        animate=False
                    )
                except Exception:  # noqa: S110
                    pass
            return

        # Plan section
        if self._plan and self._plan.get("steps"):
            plan_title = self._plan.get("plan", "Plan")
            completed = self._status == "completed" and all(
                self._plan_step_status.get(step.get("step_number")) == "done"
                for step in self._plan["steps"]
            )
            if completed:
                total = len(self._plan["steps"])
                t.append("  PLAN  ", style=f"bold {_C_MUTED}")
                t.append(f"✔ {total} steps completed\n\n", style=_C_MUTED)
            else:
                t.append("  PLAN\n", style=f"bold {_C_MUTED}")
                t.append("  ▸ ", style=f"bold {_C_TEAL}")
                t.append(f"{plan_title[:80]}\n", style=f"bold {_C_TEAL}")
                t.append("\n")

                for step in self._plan["steps"]:
                    sn = step.get("step_number", 0)
                    desc = step.get("description", "")
                    short = desc[:90]
                    if len(desc) > 90:
                        short += "…"

                    st = self._plan_step_status.get(sn, "pending")
                    if st == "done":
                        t.append("  ✔ ", style=_C_GREEN)
                        t.append(f"{sn}. {short}\n", style=_C_MUTED)
                    elif st == "failed":
                        t.append("  ✘ ", style=_C_RED)
                        t.append(f"{sn}. {short}\n", style=_C_RED)
                    elif st == "active":
                        t.append(f"  {self._spinner()} ", style=_C_PRIMARY)
                        t.append(f"{sn}. {short}\n", style=_C_TEXT)
                    else:
                        t.append("  ○ ", style=_C_MUTED)
                        t.append(f"{sn}. {short}\n", style=_C_MUTED)
                t.append("\n")

        # Current activity indicator
        if self._current_step:
            sty, msg, _detail = self._current_step
            if sty == "yellow":
                t.append(f"  {self._spinner()} ", style=_C_PRIMARY)
                t.append(f"{msg}\n\n", style=_C_DIM)
            elif sty == "teal":
                t.append(f"  {self._spinner()} ", style=_C_TEAL)
                t.append(f"{msg}\n\n", style=_C_TEAL)

        # Streaming output
        if self._is_streaming and self._streaming_text:
            text = self._filtered_streaming_text()
            text = _unescape_text(text)
            if text.strip():
                lines = text.rstrip().split("\n")
                for line in lines[-40:]:
                    t.append(f"  {line}\n", style=_C_TEXT)
            should_scroll = True

        # Final output
        if self._status == "completed" and self._final_output:
            t.append("\n")
            t.append("  ━━━ Result ━━━\n\n", style=f"bold {_C_TEAL}")
            output = _unescape_text(self._final_output)
            output = _format_json_in_text(output)
            is_json = output.lstrip()[:1] in ("{", "[", '"')
            for line in output.split("\n"):
                t.append("  ")
                if is_json:
                    _colorize_json_line(t, line)
                else:
                    t.append(line, style=_C_TEXT)
                t.append("\n")

        widget.update(t)

        if should_scroll:
            try:
                scroll = self.query_one("#scroll-area", VerticalScroll)
                if (
                    scroll.max_scroll_y <= 0
                    or scroll.scroll_y >= scroll.max_scroll_y - 50
                ):
                    scroll.scroll_end(animate=False)
            except Exception:  # noqa: S110
                pass

    # ── Log panel rendering ──────────────────────────────────

    def _render_log_panel(self) -> None:
        widget = self.query_one("#log-content", Static)
        t = Text()
        t.append("  ACTIVITY LOG", style=f"bold {_C_PRIMARY}")
        t.append("  ↑↓ navigate  enter expand/collapse\n", style=_C_MUTED)

        if not self._log_entries:
            t.append("\n  No activity yet.\n", style=_C_MUTED)
            widget.update(t)
            return

        if self._log_cursor >= len(self._log_entries):
            self._log_cursor = len(self._log_entries) - 1

        cursor_line = 0
        line_map: list[tuple[int, int, int]] = []
        now = time.time()
        for i, entry in enumerate(self._log_entries):
            entry_start_line = t.plain.count("\n")
            name = entry["tool_name"]
            status = entry["status"]
            focused = i == self._log_cursor
            expanded = i in self._log_expanded
            if focused:
                cursor_line = entry_start_line

            if status == "running" and (now - entry["start_time"]) > 120:
                entry["status"] = "timeout"
                entry["error"] = "No response received (timeout)"
                entry["duration"] = now - entry["start_time"]
                status = "timeout"
                self._log_expanded.add(i)

            arrow = "▾" if expanded else "▸"

            if focused:
                t.append("\n")
                t.append(" > ", style=_C_PRIMARY)
            else:
                t.append("\n   ", style="")

            if status == "running":
                elapsed = now - entry["start_time"]
                t.append(f"{arrow} ", style=_C_MUTED)
                t.append(f"{self._spinner()} ", style=_C_PRIMARY)
                t.append(f"{name}", style=f"bold {_C_TEXT}" if focused else _C_TEXT)
                t.append(f"  {elapsed:.0f}s\n", style=_C_MUTED)
            elif status == "success":
                t.append(f"{arrow} ", style=_C_MUTED)
                t.append("✔ ", style=_C_GREEN)
                t.append(f"{name}", style=f"bold {_C_TEXT}" if focused else _C_DIM)
                if entry.get("from_cache"):
                    t.append("  cached\n", style=_C_TEAL)
                else:
                    t.append(f"  {entry['duration']:.1f}s\n", style=_C_MUTED)
            elif status in ("error", "timeout"):
                t.append(f"{arrow} ", style=_C_MUTED)
                t.append("✘ ", style=_C_RED)
                t.append(f"{name}", style=f"bold {_C_RED}")
                dur = f"  {entry['duration']:.1f}s" if entry.get("duration") else ""
                t.append(f"{dur}\n", style=_C_MUTED)

            if not expanded:
                continue

            indent = "       "
            if entry.get("args"):
                t.append(f"{indent}Args:\n", style=_C_MUTED)
                try:
                    parsed = _json.loads(entry["args"])
                    formatted = _json.dumps(parsed, indent=2, ensure_ascii=False)
                except (ValueError, TypeError):
                    formatted = entry["args"]
                _append_highlighted(t, formatted, indent)

            if status in ("error", "timeout") and entry.get("error"):
                t.append(f"{indent}Error:\n", style=_C_RED)
                for line in str(entry["error"]).split("\n"):
                    if line.strip():
                        t.append(f"{indent}  {line}\n", style=_C_RED)

            if status == "success" and entry.get("result"):
                t.append(f"{indent}Result:\n", style=_C_TEAL)
                result_text = _unescape_text(str(entry["result"]))
                result_text = _format_json_in_text(result_text)
                total = _append_highlighted(t, result_text, indent)
                if total > 50:
                    t.append(f"{indent}  … ({total} lines total)\n", style=_C_MUTED)

            line_map.append((i, entry_start_line, t.plain.count("\n")))

        self._log_line_map = line_map
        widget.update(t)

        if self._log_scroll_needed:
            self._log_scroll_needed = False
            try:
                log_scroll = self.query_one("#log-panel", VerticalScroll)
                panel_h = log_scroll.size.height
                cursor_top = cursor_line
                cursor_bottom = cursor_line + 2
                for _idx, _start, _end in self._log_line_map:
                    if _idx == self._log_cursor:
                        cursor_bottom = _end
                        break
                visible_top = int(log_scroll.scroll_y)
                visible_bottom = visible_top + panel_h
                if cursor_top < visible_top + 1:
                    log_scroll.scroll_to(y=max(0, cursor_top - 1), animate=False)
                elif cursor_bottom > visible_bottom - 1:
                    log_scroll.scroll_to(
                        y=max(0, cursor_bottom - panel_h + 1), animate=False
                    )
            except Exception:  # noqa: S110
                pass

    def _filtered_streaming_text(self) -> str:
        if not self._streaming_text:
            return ""
        text = self._streaming_text

        # Strip plan JSON — both complete (already parsed) and in-progress
        plan_start = text.find('{"plan"')
        if plan_start >= 0:
            depth = 0
            for i in range(plan_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        text = (text[:plan_start] + text[i + 1 :]).strip()
                        break
            else:
                # Incomplete JSON — hide the partial blob
                text = text[:plan_start].strip()

        text = self._strip_step_observation_json(text)
        return _format_json_in_text(text)

    def _strip_step_observation_json(self, text: str) -> str:
        """Hide structured step-observation JSON from the live transcript."""
        if "step_completed_successfully" not in text:
            return text

        result: list[str] = []
        decoder = _json.JSONDecoder()
        i = 0
        while i < len(text):
            start = text.find("{", i)
            if start < 0:
                result.append(text[i:])
                break

            result.append(text[i:start])
            try:
                parsed, offset = decoder.raw_decode(text[start:])
            except ValueError:
                if "step_completed_successfully" in text[start:]:
                    break
                result.append(text[start])
                i = start + 1
                continue

            end = start + offset
            if self._is_step_observation_payload(parsed):
                i = end
                continue

            result.append(text[start:end])
            i = end

        return "".join(result).strip()

    @staticmethod
    def _is_step_observation_payload(payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and "step_completed_successfully" in payload
            and "key_information_learned" in payload
        )

    # ── Event helpers ───────────────────────────────────────

    def _complete_step(self, style: str, message: str, detail: str = "") -> None:
        with self._lock:
            if self._current_step:
                prev_style, prev_msg, prev_detail = self._current_step
                skip = prev_msg in (
                    "Thinking…",
                    "Generating response…",
                ) or prev_msg.startswith("⚡")
                if not skip:
                    self._timeline.append((prev_style, prev_msg, prev_detail))
            self._current_step = (style, message, detail)
            if len(self._timeline) > 20:
                self._timeline = self._timeline[-20:]

    def _replace_step(self, style: str, message: str, detail: str = "") -> None:
        """Replace current step in-place (no archive). Used for tool results."""
        with self._lock:
            self._current_step = (style, message, detail)

    def _set_step(self, style: str, message: str) -> None:
        with self._lock:
            self._current_step = (style, message, "")

    # ── Plan detection ──────────────────────────────────────

    def _try_parse_plan(self, text: str) -> None:
        stripped = text.strip()
        start = stripped.find('{"plan"')
        if start < 0:
            return
        depth = 0
        for i in range(start, len(stripped)):
            if stripped[i] == "{":
                depth += 1
            elif stripped[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = _json.loads(stripped[start : i + 1])
                        if "steps" in data and isinstance(data["steps"], list):
                            self._plan = data
                            self._plan_step_status = {
                                s["step_number"]: "pending"
                                for s in data["steps"]
                                if "step_number" in s
                            }
                            self._awaiting_replan = False
                    except (ValueError, KeyError):
                        # Best-effort parse of streamed planner output:
                        # partial or non-plan JSON is expected and ignored.
                        pass
                    return

    def _set_plan_step_status(self, step_number: int, status: str) -> None:
        """Set status for an explicit plan step reported by the planner."""
        if not self._plan or step_number not in self._plan_step_status:
            return

        self._plan_step_status[step_number] = status

    def _mark_plan_goal_achieved(self, step_number: int | None = None) -> None:
        """Collapse early-goal/skipped plan steps into completed UI state."""
        if not self._plan:
            return

        if step_number is not None:
            self._set_plan_step_status(step_number, "done")

        for sn, current in list(self._plan_step_status.items()):
            if current in ("pending", "active"):
                self._plan_step_status[sn] = "done"

    def _collapse_plan_on_task_done(self) -> None:
        """Collapse unfinished display-only plan steps once the task succeeds."""
        if not self._plan:
            return

        for sn, current in list(self._plan_step_status.items()):
            if current in ("pending", "active"):
                self._plan_step_status[sn] = "done"

    def _pop_task_state(self, event: Any) -> dict[str, Any]:
        """Return the start-time state for a completion/failure event's task.

        Tasks can run async/overlapping, so the event's task identity is
        matched against the state registered when the task started rather
        than assuming the most recently started task. Falls back to the
        current shared state for unmatched events. Caller must hold
        ``self._lock``.
        """
        task = getattr(event, "task", None)
        candidates: list[str] = []
        if task is not None:
            task_id = str(getattr(task, "id", "") or "")
            if task_id:
                candidates.append(task_id)
            desc = getattr(task, "name", "") or getattr(task, "description", "") or ""
            if desc:
                candidates.append(desc)
        event_task_name = getattr(event, "task_name", "") or ""
        if event_task_name:
            candidates.append(event_task_name)
        for key in candidates:
            state = self._task_state_by_key.pop(key, None)
            if state is not None:
                return state
        return {
            "idx": self._current_task_idx,
            "desc": self._current_task_desc,
            "agent": self._current_agent,
            "start_time": self._task_start_time,
        }

    def _prepare_for_replan(self) -> None:
        """Keep current statuses visible while allowing the next plan to replace it."""
        self._awaiting_replan = True

    def _apply_plan_refinements(self, refinements: list[str] | None) -> None:
        """Apply refined descriptions while leaving statuses as pending/done/failed."""
        if not self._plan or not refinements:
            return

        steps = self._plan.get("steps")
        if not isinstance(steps, list):
            return

        steps_by_number = {
            step.get("step_number"): step for step in steps if isinstance(step, dict)
        }
        for refinement in refinements:
            match = _REFINEMENT_RE.match(refinement)
            if not match:
                continue
            step_number = int(match.group(1))
            description = match.group(2).strip()
            step = steps_by_number.get(step_number)
            if step is not None and description:
                step["description"] = description

    def _try_parse_step_observation(self, text: str) -> bool:
        """Parse streamed observation JSON and update the exact step it names."""
        if "step_completed_successfully" not in text:
            return False

        decoder = _json.JSONDecoder()
        updated = False
        i = 0
        while i < len(text):
            start = text.find("{", i)
            if start < 0:
                break
            try:
                payload, offset = decoder.raw_decode(text[start:])
            except ValueError:
                i = start + 1
                continue

            if self._is_step_observation_payload(payload):
                step_number = self._observation_step_number(payload)
                if step_number is not None:
                    status = (
                        "done"
                        if payload.get("step_completed_successfully") is True
                        else "failed"
                    )
                    self._set_plan_step_status(step_number, status)
                    if payload.get("goal_already_achieved") is True:
                        self._mark_plan_goal_achieved(step_number)
                    updated = True
            i = start + max(offset, 1)

        return updated

    def _observation_step_number(self, payload: dict[str, Any]) -> int | None:
        raw_step_number = payload.get("step_number")
        if isinstance(raw_step_number, int):
            return raw_step_number

        searchable = " ".join(
            str(payload.get(field) or "")
            for field in ("key_information_learned", "replan_reason")
        )
        match = _STEP_NUMBER_RE.search(searchable)
        if not match:
            return None

        return int(match.group(1))

    # ── Event subscription ──────────────────────────────────

    def _register_handler(self, event_type: type, handler: Any) -> None:
        self._event_handlers.append((event_type, handler))

    def _unsubscribe(self) -> None:
        if not self._event_handlers:
            return
        try:
            from crewai.events.event_bus import crewai_event_bus

            for event_type, handler in self._event_handlers:
                crewai_event_bus.off(event_type, handler)
        except Exception:  # noqa: S110
            pass
        self._event_handlers.clear()

    def _has_running_memory_save_locked(self) -> bool:
        return any(
            entry["tool_name"] == "memory_save" and entry["status"] == "running"
            for entry in self._log_entries
        )

    def _on_memory_save_drain_elapsed(self) -> None:
        self._memory_save_drain_timer = None
        self._unsubscribe_if_no_running_memory_save()

    def _schedule_memory_save_drain_unsubscribe(self) -> bool:
        loop = getattr(self, "_loop", None)
        if loop is None:
            return False
        if getattr(self, "_thread_id", None) != threading.get_ident():
            try:
                loop.call_soon_threadsafe(self._schedule_memory_save_drain_unsubscribe)
            except RuntimeError:
                return False
            return True
        if self._memory_save_drain_timer is not None:
            self._memory_save_drain_timer.stop()
        self._memory_save_drain_timer = self.set_timer(
            _MEMORY_SAVE_DRAIN_GRACE_SECONDS,
            self._on_memory_save_drain_elapsed,
            name="memory-save-drain",
        )
        return True

    def _unsubscribe_if_no_running_memory_save(
        self, *, wait_for_queued: bool = False
    ) -> None:
        with self._lock:
            should_unsubscribe = (
                self._status
                in {
                    "completed",
                    "failed",
                }
                and not self._has_running_memory_save_locked()
            )

        if should_unsubscribe:
            if wait_for_queued and self._schedule_memory_save_drain_unsubscribe():
                return
            self._unsubscribe()

    def _subscribe(self) -> None:
        from crewai.events.event_bus import crewai_event_bus
        from crewai.events.types.crew_events import CrewKickoffStartedEvent
        from crewai.events.types.llm_events import (
            LLMCallCompletedEvent,
            LLMCallStartedEvent,
            LLMStreamChunkEvent,
        )
        from crewai.events.types.logging_events import (
            AgentLogsExecutionEvent,
            AgentLogsStartedEvent,
        )
        from crewai.events.types.observation_events import (
            GoalAchievedEarlyEvent,
            PlanRefinementEvent,
            PlanReplanTriggeredEvent,
            PlanStepCompletedEvent,
            PlanStepStartedEvent,
            StepObservationCompletedEvent,
            StepObservationFailedEvent,
            StepObservationStartedEvent,
        )
        from crewai.events.types.task_events import (
            TaskCompletedEvent,
            TaskFailedEvent,
            TaskStartedEvent,
        )
        from crewai.events.types.tool_usage_events import (
            ToolUsageErrorEvent,
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
        )

        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def on_crew_started(source: Any, event: CrewKickoffStartedEvent) -> None:
            with self._lock:
                if event.crew_name:
                    self._crew_name = event.crew_name
                    self.title = f"CrewAI — {event.crew_name}"
                self._status = "working"

        self._register_handler(CrewKickoffStartedEvent, on_crew_started)

        @crewai_event_bus.on(TaskStartedEvent)
        def on_task_started(source: Any, event: TaskStartedEvent) -> None:
            with self._lock:
                self._current_task_idx += 1
                idx = self._current_task_idx
                self._task_start_time = time.time()
                self._streaming_text = ""
                self._task_full_output = ""
                self._is_streaming = False
                self._plan = None
                self._plan_step_status = {}
                self._awaiting_replan = False

                # Tasks may run async/overlapping, so earlier active rows are
                # only marked done by their own completion events (with a
                # final sweep in _on_crew_done).
                if idx in self._task_statuses:
                    self._task_statuses[idx] = "active"

                desc = ""
                if event.task:
                    desc = getattr(event.task, "name", "") or ""
                    if not desc:
                        desc = getattr(event.task, "description", "") or ""
                if not desc and event.task_name:
                    desc = event.task_name
                self._current_task_desc = desc

                agent = getattr(source, "agent", None) if source else None
                agent_role = (getattr(agent, "role", "") or "") if agent else ""
                if agent_role:
                    self._current_agent = agent_role

                key = str(getattr(event.task, "id", "") or "") or desc
                if key:
                    self._task_state_by_key[key] = {
                        "idx": idx,
                        "desc": desc,
                        "agent": agent_role,
                        "start_time": self._task_start_time,
                    }

                self._timeline = []
                self._current_step = None
            self._set_step("yellow", "Thinking…")

        self._register_handler(TaskStartedEvent, on_task_started)

        @crewai_event_bus.on(AgentLogsStartedEvent)
        def on_agent_started(source: Any, event: AgentLogsStartedEvent) -> None:
            with self._lock:
                role = event.agent_role.split("\n")[0] if event.agent_role else ""
                if role:
                    self._current_agent = role

        self._register_handler(AgentLogsStartedEvent, on_agent_started)

        @crewai_event_bus.on(LLMCallStartedEvent)
        def on_llm_started(source: Any, event: LLMCallStartedEvent) -> None:
            with self._lock:
                self._is_streaming = False
                self._streaming_text = ""
                self._live_out_tokens = 0
                self._current_llm_text = ""
                if event.messages:
                    estimate = len(str(event.messages)) // 4
                    self._input_tokens += estimate
                    self._pending_input_estimate = estimate
            self._complete_step("yellow", "Thinking…")

        self._register_handler(LLMCallStartedEvent, on_llm_started)

        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_completed(source: Any, event: LLMCallCompletedEvent) -> None:
            with self._lock:
                self._llm_calls += 1
                self._is_streaming = False
                self._streaming_text = ""
                self._live_out_tokens = 0
                self._input_tokens -= self._pending_input_estimate
                self._pending_input_estimate = 0
                if event.usage:
                    u = event.usage
                    inp = next(
                        (
                            u[k]
                            for k in (
                                "prompt_tokens",
                                "input_tokens",
                                "prompt_token_count",
                            )
                            if u.get(k)
                        ),
                        0,
                    )
                    out = next(
                        (
                            u[k]
                            for k in (
                                "completion_tokens",
                                "output_tokens",
                                "candidates_token_count",
                            )
                            if u.get(k)
                        ),
                        0,
                    )
                    self._input_tokens += inp
                    self._output_tokens += out
                if self._current_llm_text.strip():
                    self._current_task_steps.append(
                        {
                            "type": "llm",
                            "summary": f"LLM response (call {self._llm_calls})",
                            "detail": self._current_llm_text.strip(),
                            "style": "dim",
                        }
                    )
                    self._current_llm_text = ""

        self._register_handler(LLMCallCompletedEvent, on_llm_completed)

        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_stream_chunk(source: Any, event: LLMStreamChunkEvent) -> None:
            with self._lock:
                if not self._is_streaming:
                    self._current_step = ("yellow", "Generating response…", "")
                self._is_streaming = True
                self._streaming_text += event.chunk
                self._task_full_output += event.chunk
                self._current_llm_text += event.chunk
                self._live_out_tokens += 1
                if (
                    not self._plan or self._awaiting_replan
                ) and '{"plan"' in self._streaming_text:
                    self._try_parse_plan(self._streaming_text)
                if self._plan and "step_completed_successfully" in self._streaming_text:
                    self._try_parse_step_observation(self._streaming_text)

        self._register_handler(LLMStreamChunkEvent, on_stream_chunk)

        @crewai_event_bus.on(StepObservationStartedEvent)
        def on_step_observation_started(
            source: Any, event: StepObservationStartedEvent
        ) -> None:
            with self._lock:
                self._set_plan_step_status(event.step_number, "active")

        self._register_handler(StepObservationStartedEvent, on_step_observation_started)

        @crewai_event_bus.on(StepObservationCompletedEvent)
        def on_step_observation_completed(
            source: Any, event: StepObservationCompletedEvent
        ) -> None:
            with self._lock:
                status = "done" if event.step_completed_successfully else "failed"
                self._set_plan_step_status(event.step_number, status)

        self._register_handler(
            StepObservationCompletedEvent, on_step_observation_completed
        )

        @crewai_event_bus.on(StepObservationFailedEvent)
        def on_step_observation_failed(
            source: Any, event: StepObservationFailedEvent
        ) -> None:
            with self._lock:
                # Intentionally "done", not "failed": this event means the
                # step OBSERVER failed (e.g. timeout), not the step itself,
                # and the executor continues past it. A red ✘ would wrongly
                # suggest the plan step failed.
                self._set_plan_step_status(event.step_number, "done")

        self._register_handler(StepObservationFailedEvent, on_step_observation_failed)

        @crewai_event_bus.on(PlanRefinementEvent)
        def on_plan_refinement(source: Any, event: PlanRefinementEvent) -> None:
            with self._lock:
                if event.step_number:
                    self._set_plan_step_status(event.step_number, "done")
                self._apply_plan_refinements(event.refinements)

        self._register_handler(PlanRefinementEvent, on_plan_refinement)

        @crewai_event_bus.on(PlanStepStartedEvent)
        def on_plan_step_started(source: Any, event: PlanStepStartedEvent) -> None:
            with self._lock:
                self._set_plan_step_status(event.step_number, "active")

        self._register_handler(PlanStepStartedEvent, on_plan_step_started)

        @crewai_event_bus.on(PlanStepCompletedEvent)
        def on_plan_step_completed(source: Any, event: PlanStepCompletedEvent) -> None:
            with self._lock:
                self._set_plan_step_status(
                    event.step_number,
                    "done" if event.success else "failed",
                )

        self._register_handler(PlanStepCompletedEvent, on_plan_step_completed)

        @crewai_event_bus.on(PlanReplanTriggeredEvent)
        def on_plan_replan_triggered(
            source: Any, event: PlanReplanTriggeredEvent
        ) -> None:
            with self._lock:
                self._prepare_for_replan()
                self._current_step = ("yellow", "Replanning…", event.replan_reason)

        self._register_handler(PlanReplanTriggeredEvent, on_plan_replan_triggered)

        @crewai_event_bus.on(GoalAchievedEarlyEvent)
        def on_goal_achieved_early(source: Any, event: GoalAchievedEarlyEvent) -> None:
            with self._lock:
                self._mark_plan_goal_achieved(event.step_number or None)

        self._register_handler(GoalAchievedEarlyEvent, on_goal_achieved_early)

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def on_tool_started(source: Any, event: ToolUsageStartedEvent) -> None:
            if event.tool_name in _INTERNAL_TOOL_NAMES:
                return

            with self._lock:
                self._is_streaming = False
                self._streaming_text = ""
                now = time.time()
                args_str = ""
                if event.tool_args:
                    try:
                        args_str = _json.dumps(event.tool_args, indent=2, default=str)
                    except Exception:
                        args_str = str(event.tool_args)
                for entry in self._log_entries:
                    if (
                        entry["status"] == "running"
                        and entry["tool_name"] == event.tool_name
                        and entry["args"] == (args_str or None)
                    ):
                        return
                for entry in self._log_entries:
                    if (
                        entry["status"] == "running"
                        and entry["tool_name"] != event.tool_name
                    ):
                        if entry["tool_name"] == "memory_save":
                            continue
                        entry["status"] = "timeout"
                        entry["error"] = (
                            "No result received before the next tool started"
                        )
                        entry["duration"] = now - entry["start_time"]
                plan_step_number = getattr(event, "plan_step_number", None)
                if not isinstance(plan_step_number, int):
                    plan_step_number = None
                self._current_task_steps.append(
                    {
                        "type": "tool",
                        "summary": f"Using {event.tool_name}…",
                        "detail": f"Args:\n{args_str}" if args_str else None,
                        "style": "yellow",
                        "_tool_name": event.tool_name,
                    }
                )
                self._log_entries.append(
                    {
                        "tool_name": event.tool_name,
                        "status": "running",
                        "args": args_str or None,
                        "result": None,
                        "error": None,
                        "start_time": time.time(),
                        "duration": None,
                        "task_idx": self._current_task_idx,
                        "plan_step_number": plan_step_number,
                        "event_id": event.event_id,
                    }
                )
            self._complete_step("teal", f"⚡ {event.tool_name}…")

        self._register_handler(ToolUsageStartedEvent, on_tool_started)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def on_tool_finished(source: Any, event: ToolUsageFinishedEvent) -> None:
            if event.tool_name in _INTERNAL_TOOL_NAMES:
                return

            with self._lock:
                if event.output is not None:
                    out = event.output
                    if isinstance(out, (dict, list)):
                        try:
                            result_str = _json.dumps(out, indent=2, ensure_ascii=False)[
                                :5000
                            ]
                        except (TypeError, ValueError):
                            result_str = str(out)[:5000]
                    else:
                        result_str = str(out)[:5000]
                else:
                    result_str = "No output"
                for step in reversed(self._current_task_steps):
                    if (
                        step.get("_tool_name") == event.tool_name
                        and step["type"] == "tool"
                    ):
                        existing = step.get("detail") or ""
                        step["detail"] = (
                            f"{existing}\n\nResult:\n{result_str}"
                            if existing
                            else f"Result:\n{result_str}"
                        )
                        step["summary"] = f"✔ {event.tool_name}"
                        step["style"] = "green"
                        break
                from_cache = getattr(event, "from_cache", False)
                for entry in reversed(self._log_entries):
                    if entry["tool_name"] == event.tool_name and (
                        entry["status"] == "running"
                        or (entry["status"] == "success" and entry["result"] is None)
                    ):
                        entry["status"] = "success"
                        entry["result"] = result_str
                        entry["duration"] = time.time() - entry["start_time"]
                        entry["from_cache"] = from_cache
                        break
            self._replace_step("green", f"✔ {event.tool_name}")

        self._register_handler(ToolUsageFinishedEvent, on_tool_finished)

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def on_tool_error(source: Any, event: ToolUsageErrorEvent) -> None:
            if event.tool_name in _INTERNAL_TOOL_NAMES:
                return

            error_text = str(event.error)[:200] if event.error else ""
            with self._lock:
                for step in reversed(self._current_task_steps):
                    if (
                        step.get("_tool_name") == event.tool_name
                        and step["type"] == "tool"
                    ):
                        existing = step.get("detail") or ""
                        step["detail"] = (
                            f"{existing}\n\nError:\n{event.error}"
                            if existing
                            else f"Error:\n{event.error}"
                        )
                        step["summary"] = f"✘ {event.tool_name}"
                        step["style"] = "red"
                        break
                for idx, entry in reversed(list(enumerate(self._log_entries))):
                    if entry["tool_name"] == event.tool_name and (
                        entry["status"] == "running"
                        or (entry["status"] == "success" and entry["result"] is None)
                    ):
                        entry["status"] = "error"
                        entry["error"] = str(event.error) if event.error else None
                        entry["duration"] = time.time() - entry["start_time"]
                        self._log_expanded.add(idx)
                        break
            self._replace_step("red", f"✘ {event.tool_name}", error_text)

        self._register_handler(ToolUsageErrorEvent, on_tool_error)

        from crewai.events.types.memory_events import (
            MemoryRetrievalCompletedEvent,
            MemoryRetrievalFailedEvent,
            MemoryRetrievalStartedEvent,
            MemorySaveCompletedEvent,
            MemorySaveFailedEvent,
            MemorySaveStartedEvent,
        )

        def is_nested_save_to_memory_event(event: Any) -> bool:
            if event.parent_event_id is None:
                return False
            state = crewai_event_bus.runtime_state
            if state is None:
                return False
            parent_node = state.event_record.nodes.get(event.parent_event_id)
            parent_event = getattr(parent_node, "event", None)
            return getattr(
                parent_event, "type", None
            ) == "tool_usage_started" and _is_save_to_memory_tool(
                getattr(parent_event, "tool_name", None)
            )

        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_memory_save_started(source: Any, event: MemorySaveStartedEvent) -> None:
            with self._lock:
                if is_nested_save_to_memory_event(event):
                    self._suppressed_memory_save_event_ids.add(event.event_id)
                    return
                for entry in reversed(self._log_entries):
                    if (
                        _is_save_to_memory_tool(entry["tool_name"])
                        and entry.get("event_id") == event.parent_event_id
                    ):
                        self._suppressed_memory_save_event_ids.add(event.event_id)
                        return
                for entry in reversed(self._log_entries):
                    if (
                        entry["tool_name"] == "memory_save"
                        and entry.get("started_event_id") == event.event_id
                    ):
                        entry["args"] = _truncate_log_text(
                            event.value, _LOG_ARGS_TEXT_LIMIT
                        )
                        return
                self._log_entries.append(
                    {
                        "tool_name": "memory_save",
                        "status": "running",
                        "args": _truncate_log_text(event.value, _LOG_ARGS_TEXT_LIMIT),
                        "result": None,
                        "error": None,
                        "start_time": time.time(),
                        "duration": None,
                        "task_idx": self._current_task_idx,
                        "event_id": event.event_id,
                    }
                )

        self._register_handler(MemorySaveStartedEvent, on_memory_save_started)

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_memory_save_completed(
            source: Any, event: MemorySaveCompletedEvent
        ) -> None:
            with self._lock:
                if (
                    event.started_event_id in self._suppressed_memory_save_event_ids
                    or is_nested_save_to_memory_event(event)
                ):
                    if event.started_event_id is not None:
                        self._suppressed_memory_save_event_ids.discard(
                            event.started_event_id
                        )
                else:
                    for entry in reversed(self._log_entries):
                        has_started_event_match = (
                            event.started_event_id is not None
                            and (
                                entry.get("event_id") == event.started_event_id
                                or entry.get("started_event_id")
                                == event.started_event_id
                            )
                        )
                        has_running_event_without_id = (
                            event.started_event_id is None
                            and entry["status"] == "running"
                        )
                        if entry["tool_name"] == "memory_save" and (
                            has_running_event_without_id or has_started_event_match
                        ):
                            entry["status"] = "success"
                            entry["duration"] = event.save_time_ms / 1000
                            entry["result"] = _truncate_log_text(
                                event.value, _LOG_RESULT_TEXT_LIMIT
                            )
                            entry["error"] = None
                            entry["started_event_id"] = event.started_event_id
                            break
                    else:
                        self._log_entries.append(
                            {
                                "tool_name": "memory_save",
                                "status": "success",
                                "args": None,
                                "result": _truncate_log_text(
                                    event.value, _LOG_RESULT_TEXT_LIMIT
                                ),
                                "error": None,
                                "start_time": time.time(),
                                "duration": event.save_time_ms / 1000,
                                "task_idx": self._current_task_idx,
                                "started_event_id": event.started_event_id,
                            }
                        )

            self._unsubscribe_if_no_running_memory_save(wait_for_queued=True)

        self._register_handler(MemorySaveCompletedEvent, on_memory_save_completed)

        @crewai_event_bus.on(MemorySaveFailedEvent)
        def on_memory_save_failed(source: Any, event: MemorySaveFailedEvent) -> None:
            with self._lock:
                if (
                    event.started_event_id in self._suppressed_memory_save_event_ids
                    or is_nested_save_to_memory_event(event)
                ):
                    if event.started_event_id is not None:
                        self._suppressed_memory_save_event_ids.discard(
                            event.started_event_id
                        )
                else:
                    for idx, entry in reversed(list(enumerate(self._log_entries))):
                        has_started_event_match = (
                            event.started_event_id is not None
                            and (
                                entry.get("event_id") == event.started_event_id
                                or entry.get("started_event_id")
                                == event.started_event_id
                            )
                        )
                        has_running_event_without_id = (
                            event.started_event_id is None
                            and entry["status"] == "running"
                        )
                        if entry["tool_name"] == "memory_save" and (
                            has_running_event_without_id or has_started_event_match
                        ):
                            entry["status"] = "error"
                            entry["error"] = event.error
                            entry["duration"] = time.time() - entry["start_time"]
                            entry["started_event_id"] = event.started_event_id
                            self._log_expanded.add(idx)
                            break
                    else:
                        self._log_entries.append(
                            {
                                "tool_name": "memory_save",
                                "status": "error",
                                "args": _truncate_log_text(
                                    event.value, _LOG_ARGS_TEXT_LIMIT
                                ),
                                "result": None,
                                "error": event.error,
                                "start_time": time.time(),
                                "duration": 0,
                                "task_idx": self._current_task_idx,
                                "started_event_id": event.started_event_id,
                            }
                        )
                        self._log_expanded.add(len(self._log_entries) - 1)

            self._unsubscribe_if_no_running_memory_save(wait_for_queued=True)

        self._register_handler(MemorySaveFailedEvent, on_memory_save_failed)

        @crewai_event_bus.on(MemoryRetrievalStartedEvent)
        def on_memory_retrieval_started(
            source: Any, event: MemoryRetrievalStartedEvent
        ) -> None:
            with self._lock:
                self._log_entries.append(
                    {
                        "tool_name": "memory_recall",
                        "status": "running",
                        "args": None,
                        "result": None,
                        "error": None,
                        "start_time": time.time(),
                        "duration": None,
                        "task_idx": self._current_task_idx,
                    }
                )

        self._register_handler(MemoryRetrievalStartedEvent, on_memory_retrieval_started)

        @crewai_event_bus.on(MemoryRetrievalCompletedEvent)
        def on_memory_retrieval_completed(
            source: Any, event: MemoryRetrievalCompletedEvent
        ) -> None:
            with self._lock:
                for entry in reversed(self._log_entries):
                    if (
                        entry["tool_name"] == "memory_recall"
                        and entry["status"] == "running"
                    ):
                        entry["status"] = "success"
                        entry["duration"] = event.retrieval_time_ms / 1000
                        content = event.memory_content or ""
                        if content:
                            entry["result"] = content[:3000]
                        break

        self._register_handler(
            MemoryRetrievalCompletedEvent, on_memory_retrieval_completed
        )

        @crewai_event_bus.on(MemoryRetrievalFailedEvent)
        def on_memory_retrieval_failed(
            source: Any, event: MemoryRetrievalFailedEvent
        ) -> None:
            with self._lock:
                for idx, entry in enumerate(self._log_entries):
                    if (
                        entry["tool_name"] == "memory_recall"
                        and entry["status"] == "running"
                    ):
                        entry["status"] = "error"
                        entry["error"] = event.error
                        entry["duration"] = 0
                        self._log_expanded.add(idx)
                        break

        self._register_handler(MemoryRetrievalFailedEvent, on_memory_retrieval_failed)

        @crewai_event_bus.on(AgentLogsExecutionEvent)
        def on_agent_execution(source: Any, event: AgentLogsExecutionEvent) -> None:
            from crewai.agents.parser import AgentAction, AgentFinish

            if isinstance(event.formatted_answer, AgentAction):
                self._complete_step("cyan", f"→ {event.formatted_answer.tool}")
            elif isinstance(event.formatted_answer, AgentFinish):
                self._complete_step("green", "✔ Agent finished")

        self._register_handler(AgentLogsExecutionEvent, on_agent_execution)

        @crewai_event_bus.on(TaskCompletedEvent)
        def on_task_completed(source: Any, event: TaskCompletedEvent) -> None:
            now = time.time()
            with self._lock:
                state = self._pop_task_state(event)
                idx = state["idx"]
                self._task_statuses[idx] = "done"
                elapsed = now - state["start_time"]
                # The shared stream fields (steps, timeline, streamed output)
                # belong to the most recently started task. Only consume and
                # reset them when that is the task completing — an earlier
                # task finishing out of order must not steal or clear the
                # current task's stream.
                is_current = idx == self._current_task_idx
                output = getattr(event.output, "raw", "") or ""

                if is_current:
                    self._collapse_plan_on_task_done()

                    if self._current_llm_text.strip():
                        self._current_task_steps.append(
                            {
                                "type": "llm",
                                "summary": "Final response",
                                "detail": self._current_llm_text.strip(),
                                "style": "green",
                            }
                        )
                        self._current_llm_text = ""

                    steps = list(self._current_task_steps)
                    self._current_task_steps = []
                    timeline = list(self._timeline)
                    output = self._task_full_output or output

                    self._is_streaming = False
                    self._streaming_text = ""
                    self._task_full_output = ""
                    self._timeline = []
                    self._current_step = None
                else:
                    steps = []
                    timeline = []

                self._task_logs.append(
                    {
                        "idx": idx,
                        "desc": state["desc"] or "Task",
                        "agent": state["agent"],
                        "elapsed": elapsed,
                        "timeline": timeline,
                        "steps": steps,
                        "output": output,
                    }
                )

        self._register_handler(TaskCompletedEvent, on_task_completed)

        @crewai_event_bus.on(TaskFailedEvent)
        def on_task_failed(source: Any, event: TaskFailedEvent) -> None:
            now = time.time()
            with self._lock:
                state = self._pop_task_state(event)
                idx = state["idx"]
                self._task_statuses[idx] = "failed"
                is_current = idx == self._current_task_idx

                error_step = {
                    "type": "error",
                    "summary": f"✘ Failed: {event.error[:100]}",
                    "detail": event.error,
                    "style": "red",
                }
                if is_current:
                    self._current_task_steps.append(error_step)
                    steps = list(self._current_task_steps)
                    self._current_task_steps = []
                    timeline = list(self._timeline)
                    output = self._task_full_output
                else:
                    steps = [error_step]
                    timeline = []
                    output = ""

                self._task_logs.append(
                    {
                        "idx": idx,
                        "desc": state["desc"] or "Task",
                        "agent": state["agent"],
                        "elapsed": now - state["start_time"],
                        "timeline": timeline,
                        "steps": steps,
                        "output": output,
                        "error": event.error,
                    }
                )
            self._complete_step(
                "red", f"✘ Failed: {event.error[:50]}", event.error[:200]
            )

        self._register_handler(TaskFailedEvent, on_task_failed)
