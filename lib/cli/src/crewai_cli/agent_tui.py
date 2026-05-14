"""Textual TUI for conversational multi-agent interaction.

Launched by ``crewai run`` when agents/ directory contains agent definitions.
Features: Common Room, @mention autocomplete, inline thinking animation,
token/time metadata, conversation history persistence.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from rich.markup import escape as _rich_escape
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    OptionList,
    RadioButton,
    RadioSet,
    Static,
    TabPane,
    TabbedContent,
    TextArea,
)

from crewai_cli.create_agent import _strip_jsonc
from crewai_cli.utils import load_env_vars


try:
    from textual.suggester import Suggester

    class AgentSuggester(Suggester):
        """Autocomplete @agent_name mentions in the input."""

        def __init__(self, agent_names: list[str]) -> None:
            super().__init__(use_cache=False)
            self._names = agent_names

        async def get_suggestion(self, value: str) -> str | None:
            at_idx = value.rfind("@")
            if at_idx == -1:
                return None
            after = value[at_idx + 1 :]
            if not after or " " in after:
                return None
            lower = after.lower()
            for name in self._names:
                if name.lower().startswith(lower) and name.lower() != lower:
                    return value[: at_idx + 1] + name + " "
            return None

except ImportError:
    AgentSuggester = None  # type: ignore[assignment,misc]


class ChatTextArea(TextArea):
    """Multiline chat input: Enter submits, Shift+Enter inserts newline, Tab completes @mentions."""

    BINDINGS = [
        Binding("enter", "submit", "Send", show=False),
    ]

    class Submitted(Message):
        """Posted when the user presses Enter to submit."""

        def __init__(self, text_area: ChatTextArea, value: str) -> None:
            super().__init__()
            self.text_area = text_area
            self.value = value

    class MentionChanged(Message):
        """Posted when @mention autocomplete state changes."""

        def __init__(self, prefix: str, matches: list[str]) -> None:
            super().__init__()
            self.prefix = prefix
            self.matches = matches

    def __init__(self, agent_names: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent_names = agent_names or []
        self._last_mention_prefix: str | None = None

    def _get_mention_context(self) -> tuple[int, int, str] | None:
        """Return (row, at_col, partial) if cursor is inside an @mention."""
        row, col = self.cursor_location
        lines = self.text.split("\n")
        if row >= len(lines):
            return None
        line_to_cursor = lines[row][:col]
        at_idx = line_to_cursor.rfind("@")
        if at_idx == -1:
            return None
        after = line_to_cursor[at_idx + 1 :]
        if " " in after:
            return None
        return row, at_idx, after.lower()

    def _get_matches(self, prefix: str) -> list[str]:
        if not prefix:
            return self._agent_names[:]
        return [n for n in self._agent_names if n.lower().startswith(prefix)]

    def _emit_mention_state(self) -> None:
        ctx = self._get_mention_context()
        if ctx is None:
            if self._last_mention_prefix is not None:
                self._last_mention_prefix = None
                self.post_message(self.MentionChanged("", []))
            return
        _, _, prefix = ctx
        if prefix != self._last_mention_prefix:
            self._last_mention_prefix = prefix
            matches = self._get_matches(prefix)
            self.post_message(self.MentionChanged(prefix, matches))

    def action_submit(self) -> None:
        text = self.text
        self.clear()
        self._last_mention_prefix = None
        self.post_message(self.Submitted(self, text))
        self.post_message(self.MentionChanged("", []))

    def action_complete(self) -> None:
        """Complete the current @mention with Tab."""
        ctx = self._get_mention_context()
        if ctx is None:
            return
        row, at_col, prefix = ctx
        matches = self._get_matches(prefix)
        if not matches:
            return
        _, col = self.cursor_location
        self.replace(matches[0] + " ", start=(row, at_col + 1), end=(row, col))
        self._last_mention_prefix = None
        self.post_message(self.MentionChanged("", []))

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "shift+enter":
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "enter":
            event.prevent_default()
            self.action_submit()
            return
        if event.key == "tab":
            event.prevent_default()
            self.action_complete()
            return
        if event.key == "escape":
            self._last_mention_prefix = None
            self.post_message(self.MentionChanged("", []))
        await super()._on_key(event)
        self._emit_mention_state()


_CORAL = "#eb6658"
_TEAL = "#1F7982"
_BG = "#1a1a1a"
_BG_PANEL = "#222222"
_BG_MSG_USER = "#2a2a2a"
_BG_MSG_AGENT = "#252525"
_DIM = "#777777"
_ROOM_PREFIX = "__room__"
_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _safe_render(text: str) -> str:
    """Escape Rich markup in text so square brackets are displayed literally."""
    return _rich_escape(text)


def _load_agents(agents_dir: Path) -> list[dict[str, Any]]:
    """Load all agent definitions from agents/ directory."""
    agents: list[dict[str, Any]] = []
    for ext in ("*.json", "*.jsonc"):
        for path in sorted(agents_dir.glob(ext)):
            try:
                raw = path.read_text(encoding="utf-8")
                defn = json.loads(_strip_jsonc(raw))
                defn["_path"] = str(path)
                agents.append(defn)
            except Exception:
                pass
    return agents


def _load_config(base: Path) -> dict[str, Any]:
    """Load project config.json."""
    config_path = base / "config.json"
    if not config_path.exists():
        return {"rooms": {"common": {"agents": [], "engagement": "organic"}}}
    try:
        raw = config_path.read_text(encoding="utf-8")
        return json.loads(_strip_jsonc(raw))  # type: ignore[no-any-return]
    except Exception:
        return {"rooms": {"common": {"agents": [], "engagement": "organic"}}}


def _history_dir() -> Path:
    return Path.cwd() / ".crewai" / "tui_history"


# ── Widgets ────────────────────────────────────────────────────


class ChatBubble(Static):
    """A styled chat message bubble."""


_STATE_ICONS = {
    "recalling": "🧠",
    "dreaming": "💭",
    "planning": "📋",
    "thinking": "💡",
    "using_tool": "🔧",
    "delegating": "🤝",
}


class ThinkingIndicator(Static):
    """Animated thinking spinner with step-by-step progress log."""

    _frame: int = 0

    def __init__(self, agent_name: str) -> None:
        super().__init__()
        self._agent_name = agent_name
        self._current_status = "starting…"
        self._steps: list[str] = []
        self._tokens = ""
        self._prev_input: int = 0
        self._prev_output: int = 0
        self._step_start: float = time.monotonic()

    def update_status(
        self, state: str, detail: str | None, input_tokens: int, output_tokens: int
    ) -> None:
        label = detail or state or "working…"
        # Mark the previous step as done (skip the initial placeholder,
        # but keep its creation timestamp so the first real step inherits it)
        if self._current_status and self._current_status != "starting…":
            step_in = input_tokens - self._prev_input
            step_out = output_tokens - self._prev_output
            step_elapsed = time.monotonic() - self._step_start
            meta_parts: list[str] = []
            if step_in or step_out:
                meta_parts.append(f"↑{step_in:,} ↓{step_out:,}")
            if step_elapsed >= 0.1:
                meta_parts.append(f"{step_elapsed:.1f}s")
            meta = " · ".join(meta_parts)
            suffix = f" ({meta})" if meta else ""
            done_line = f"  [{_DIM}]✓ {_safe_render(self._current_status)}{suffix}[/]"
            if not any(self._current_status in s for s in self._steps):
                self._steps.append(done_line)
            if len(self._steps) > 6:
                self._steps = self._steps[-6:]
        self._current_status = label
        self._prev_input = input_tokens
        self._prev_output = output_tokens
        self._step_start = time.monotonic()
        if input_tokens or output_tokens:
            self._tokens = f"[{_DIM}]↑{input_tokens:,} ↓{output_tokens:,}[/]"
        self._render_frame()

    @property
    def status_text(self) -> str:
        return self._current_status

    @status_text.setter
    def status_text(self, value: str) -> None:
        self._current_status = value
        self._render_frame()

    def on_mount(self) -> None:
        self._render_frame()
        self.set_interval(0.12, self._tick)

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(_SPINNER)
        self._render_frame()

    def _render_frame(self) -> None:
        ch = _SPINNER[self._frame]
        lines: list[str] = []
        for step in self._steps:
            lines.append(step)
        status_esc = _safe_render(self._current_status)
        current = f"[{_CORAL}]{ch}[/] [{_DIM}]{self._agent_name}[/] {status_esc}"
        if self._tokens:
            current += f"  {self._tokens}"
        lines.append(current)
        content = "\n".join(lines)
        try:
            self.update(content)
        except Exception:
            self.update(_safe_render(content))


class CreateRoomScreen(ModalScreen[dict[str, Any] | None]):
    """Modal form for creating a new room."""

    CSS = f"""
    CreateRoomScreen {{
        align: center middle;
    }}
    #room-form {{
        width: 56;
        max-height: 80%;
        background: {_BG_PANEL};
        border: tall {_TEAL};
        padding: 1 2;
    }}
    #room-form Label {{
        margin: 1 0 0 0;
        color: {_DIM};
    }}
    #room-form Input {{
        margin: 0 0 1 0;
    }}
    #room-form .form-section {{
        height: auto;
        margin: 0 0 1 0;
    }}
    #room-form RadioSet {{
        margin: 0;
    }}
    #room-form Button {{
        margin: 1 1 0 0;
    }}
    """

    def __init__(self, agent_names: list[str], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._agent_names = agent_names

    def compose(self) -> ComposeResult:
        with Vertical(id="room-form"):
            yield Label(f"[bold {_CORAL}]Create Room[/]")
            yield Label("Name")
            yield Input(placeholder="e.g. engineering", id="room-name-input")
            yield Label("Agents")
            with Vertical(classes="form-section"):
                for name in self._agent_names:
                    yield Checkbox(name, value=True, id=f"cb-{name}")
            yield Label("Engagement")
            with RadioSet(id="engagement-radio"):
                yield RadioButton(
                    "Organic — agents auto-respond", value=True, id="radio-organic"
                )
                yield RadioButton("Tagged — @mention required", id="radio-tagged")
            with Horizontal():
                yield Button("Create", variant="primary", id="btn-create-room")
                yield Button("Cancel", id="btn-cancel-room")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-cancel-room":
            self.dismiss(None)
            return
        if event.button.id == "btn-create-room":
            name_input = self.query_one("#room-name-input", Input)
            name = name_input.value.strip().lower().replace(" ", "-")
            if not name:
                name_input.focus()
                return
            agents = [
                n
                for n in self._agent_names
                if self.query_one(f"#cb-{n}", Checkbox).value
            ]
            radio = self.query_one("#engagement-radio", RadioSet)
            engagement = "organic" if radio.pressed_index == 0 else "tagged"
            self.dismiss({"name": name, "agents": agents, "engagement": engagement})


# ── Main TUI ──────────────────────────────────────────────────


class AgentTUI(App[None]):
    """Multi-agent conversational TUI with Common Room support."""

    TITLE = "CrewAI Agents"
    SUB_TITLE = "Common Room"

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
    ]

    CSS = f"""
    Screen {{
        background: {_BG};
    }}
    Header {{
        background: {_CORAL};
        color: white;
    }}
    Footer {{
        background: {_BG_PANEL};
        color: {_DIM};
    }}
    Footer > .footer-key--key {{
        background: {_TEAL};
        color: white;
    }}

    #main-layout {{
        height: 1fr;
    }}

    #sidebar {{
        width: 42;
        min-width: 42;
        background: {_BG_PANEL};
        border-right: vkey #444444;
        overflow-x: hidden;
    }}
    #sidebar-tabs {{
        height: 1fr;
    }}
    #sidebar ContentSwitcher {{
        background: {_BG_PANEL};
        height: 1fr;
    }}
    #sidebar TabPane {{
        padding: 0;
    }}
    #sidebar Tabs {{
        background: {_BG};
    }}
    #sidebar Tab {{
        background: {_BG};
        color: {_DIM};
        padding: 0 1;
    }}
    #sidebar Tab.-active {{
        background: {_BG_PANEL};
        color: {_CORAL};
    }}
    #sidebar Tab:hover {{
        color: white;
    }}
    #sidebar Underline > .underline--bar {{
        color: {_TEAL};
        background: {_BG};
    }}
    .sidebar-label {{
        padding: 1 1 0 1;
        color: {_DIM};
        text-style: bold;
        height: auto;
    }}
    #room-list {{
        height: auto;
        max-height: 40%;
        padding: 0 1;
        background: {_BG_PANEL};
    }}
    #room-list > .option-list--option-highlighted {{
        background: {_TEAL};
        color: white;
    }}
    #room-list > .option-list--option {{
        padding: 0 1;
    }}
    #btn-new-room {{
        margin: 0 1 1 1;
        width: 100%;
        background: {_BG};
        color: {_TEAL};
        border: tall #333333;
        min-height: 1;
        height: 3;
    }}
    #btn-new-room:hover {{
        background: {_TEAL};
        color: white;
    }}
    #agent-list {{
        height: 1fr;
        padding: 0 1;
        background: {_BG_PANEL};
    }}
    #agent-list > .option-list--option-highlighted {{
        background: {_TEAL};
        color: white;
    }}
    #agent-list > .option-list--option {{
        padding: 0 1;
    }}
    #memory-scope-label {{
        padding: 1 1 0 1;
        color: {_DIM};
        height: auto;
    }}
    #btn-memory {{
        margin: 1;
        width: 100%;
        background: {_BG};
        color: {_CORAL};
        border: tall {_TEAL};
    }}
    #btn-memory:hover {{
        background: {_TEAL};
        color: white;
    }}
    #chat-area {{
        width: 1fr;
    }}
    #chat-scroll {{
        height: 1fr;
        padding: 1 2;
        overflow-y: auto;
    }}
    #input-row {{
        height: auto;
        max-height: 10;
        min-height: 4;
        padding: 0 1;
        background: {_BG_PANEL};
        border-top: solid #333333;
    }}
    #chat-input {{
        width: 100%;
        min-height: 3;
        max-height: 8;
        border: tall #333333;
    }}
    #chat-input:focus {{
        border: tall {_CORAL};
    }}

    .user-bubble {{
        background: {_BG_MSG_USER};
        padding: 1 2;
        margin: 1 0 1 6;
    }}
    .agent-bubble {{
        background: {_BG_MSG_AGENT};
        padding: 1 2;
        margin: 1 6 1 0;
    }}
    .system-bubble {{
        color: {_DIM};
        padding: 0 2;
        margin: 1 0 1 0;
        text-align: center;
    }}

    ThinkingIndicator {{
        padding: 0 2;
        margin: 0 0 0 0;
        height: auto;
    }}
    #completion-hint {{
        display: none;
        height: auto;
        max-height: 2;
        padding: 0 2;
        margin: 0 1;
        background: #333333;
        color: {_TEAL};
    }}
    """

    def __init__(
        self,
        agents_dir: Path,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._agents_dir = agents_dir
        self._config = config or {}
        self._config_path = Path.cwd() / "config.json"
        self._agent_defs: list[dict[str, Any]] = []
        self._agent_names: list[str] = []
        self._agent_instances: dict[str, Any] = {}
        # Rooms: {room_key: {"name": display_name, "agents": [...], "engagement": "organic"|"tagged"}}
        self._rooms: dict[str, dict[str, Any]] = {}
        self._current_room: str = ""
        # (sender, content, metadata) tuples keyed by room
        self._chat_histories: dict[str, list[tuple[str, str, str]]] = {}
        self._processing = False
        self._last_active_agent: str | None = None
        self._last_agent_error: str = ""
        self._scheduler: Any = None

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-layout"):
            with Vertical(id="sidebar"):
                with TabbedContent(id="sidebar-tabs"):
                    with TabPane("Chat", id="tab-agents"):
                        yield Static("ROOMS", classes="sidebar-label")
                        yield OptionList(id="room-list")
                        yield Button("+ New Room", id="btn-new-room", variant="default")
                        yield Static("AGENTS", classes="sidebar-label")
                        yield OptionList(id="agent-list")
                    with TabPane("Memory", id="tab-memory"):
                        yield Static(
                            "Click below to open the memory browser.",
                            id="memory-scope-label",
                        )
                        yield Button(
                            "Open Memory Browser", id="btn-memory", variant="default"
                        )
            with Vertical(id="chat-area"):
                yield VerticalScroll(id="chat-scroll")
                yield Static("", id="completion-hint")
                with Horizontal(id="input-row"):
                    yield ChatTextArea(
                        id="chat-input",
                        show_line_numbers=False,
                        theme="css",
                        soft_wrap=True,
                    )
        yield Footer()

    def _room_key(self, name: str) -> str:
        return f"{_ROOM_PREFIX}{name}"

    def _is_room(self, key: str) -> bool:
        return key.startswith(_ROOM_PREFIX)

    def _room_engagement(self, room_key: str) -> str:
        if room_key in self._rooms:
            return str(self._rooms[room_key].get("engagement", "organic"))
        return "organic"

    def _room_agents(self, room_key: str) -> list[str]:
        if room_key in self._rooms:
            return list(self._rooms[room_key].get("agents", self._agent_names[:]))
        return self._agent_names[:]

    def on_mount(self) -> None:
        self._agent_defs = _load_agents(self._agents_dir)
        self._agent_names = [
            d.get("name", d.get("role", "unnamed")) for d in self._agent_defs
        ]

        # Load rooms from config
        rooms_cfg = self._config.get("rooms", {})
        for room_name, room_data in rooms_cfg.items():
            key = self._room_key(room_name)
            cfg_agents = room_data.get("agents", [])
            self._rooms[key] = {
                "name": room_name,
                "agents": cfg_agents if cfg_agents else self._agent_names[:],
                "engagement": room_data.get("engagement", "organic"),
            }

        # Ensure at least "common" room exists
        common_key = self._room_key("common")
        if common_key not in self._rooms:
            self._rooms[common_key] = {
                "name": "common",
                "agents": self._agent_names[:],
                "engagement": "organic",
            }

        self._current_room = common_key

        # Subscribe to status update events from the executor
        self._status_listener = None
        try:
            from crewai.events.event_bus import CrewAIEventsBus

            bus = CrewAIEventsBus()
        except Exception:
            bus = None

        if bus is not None:
            try:
                from crewai.new_agent.events import NewAgentStatusUpdateEvent

                @bus.on(NewAgentStatusUpdateEvent)
                def _on_status_update(source: Any, event: Any) -> None:
                    self.call_from_thread(self._handle_status_update, source, event)

                self._status_listener = _on_status_update
            except Exception:
                pass

        if not self._agent_defs:
            self._mount_sys("No agents found. Run: crewai create agent <name>")
            return

        # Populate rooms list
        room_list = self.query_one("#room-list", OptionList)
        for key in self._rooms:
            display = self._rooms[key]["name"].replace("-", " ").title()
            engagement = self._rooms[key]["engagement"]
            n_agents = len(self._rooms[key]["agents"])
            room_list.add_option(f"◆ {display}  [{_DIM}]{engagement} · {n_agents}[/]")
        room_list.highlighted = 0

        # Populate agents list (DM entries)
        agent_list = self.query_one("#agent-list", OptionList)
        for defn in self._agent_defs:
            name = defn.get("name", "unnamed")
            role = defn.get("role", "")
            label = f"  {name}"
            if role:
                trunc = role[:18] + "…" if len(role) > 18 else role
                label += f" · {trunc}"
            agent_list.add_option(label)

        self._update_subtitle()
        self._update_placeholder()
        self._load_history_from_disk()
        self._render_chat()
        chat_input = self.query_one("#chat-input", ChatTextArea)
        chat_input._agent_names = self._agent_names
        chat_input.focus()

        try:
            from crewai.new_agent.scheduler import TaskScheduler

            self._scheduler = TaskScheduler()
            self._scheduler.set_callback(self._on_scheduled_task_due)
            self._scheduler.start()
        except Exception:
            pass

    def _update_subtitle(self) -> None:
        if self._is_room(self._current_room):
            info = self._rooms.get(self._current_room, {})
            display = info.get("name", "room").replace("-", " ").title()
            self.sub_title = display
        else:
            self.sub_title = f"Chat with {self._current_room}"

    def _update_placeholder(self) -> None:
        chat_input = self.query_one("#chat-input", ChatTextArea)
        if self._is_room(self._current_room):
            engagement = self._room_engagement(self._current_room)
            if engagement == "organic":
                chat_input.placeholder = (
                    "Type a message — agents will respond automatically"
                )
            else:
                chat_input.placeholder = "Use @agent_name to direct your message"
        else:
            chat_input.placeholder = f"Message {self._current_room}"

    def _on_scheduled_task_due(self, task: Any) -> str:
        """Callback fired by the scheduler when a task comes due."""
        agent_name = getattr(task, "agent_name", "")
        description = getattr(task, "description", "")
        if not agent_name or not description:
            return "skipped — missing agent or description"

        agent = self._get_or_create_agent(agent_name)
        if agent is None:
            return f"agent '{agent_name}' not found"

        try:
            resp = agent.message(f"[Scheduled task] {description}")
            content = getattr(resp, "content", str(resp))
            self.call_from_thread(
                self._mount_bubble,
                agent_name,
                f"[Scheduled] {content}",
                f"task: {getattr(task, 'id', '?')}",
            )
            return content[:200]
        except Exception as e:
            self.call_from_thread(
                self._mount_sys,
                f"Scheduled task '{getattr(task, 'id', '?')}' failed: {e}",
            )
            return str(e)

    # ── Sidebar navigation ──

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        if event.option_list.id == "room-list":
            room_keys = list(self._rooms.keys())
            idx = event.option_index
            if 0 <= idx < len(room_keys):
                self._current_room = room_keys[idx]
                # Deselect agent list
                try:
                    self.query_one("#agent-list", OptionList).highlighted = None
                except Exception:
                    pass
                self._update_subtitle()
                self._update_placeholder()
                self._render_chat()
        elif event.option_list.id == "agent-list":
            idx = event.option_index
            if 0 <= idx < len(self._agent_names):
                self._current_room = self._agent_names[idx]
                # Deselect room list
                try:
                    self.query_one("#room-list", OptionList).highlighted = None
                except Exception:
                    pass
                self._update_subtitle()
                self._update_placeholder()
                self._render_chat()

    # ── @mention autocomplete hint ──

    def on_chat_text_area_mention_changed(
        self, event: ChatTextArea.MentionChanged
    ) -> None:
        try:
            hint = self.query_one("#completion-hint", Static)
        except Exception:
            return
        if not event.matches:
            hint.display = False
            return
        names = "  ".join(f"@{n}" for n in event.matches[:6])
        hint.update(f"Tab to complete:  {names}")
        hint.display = True

    # ── Message routing ──

    async def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        if event.text_area.id != "chat-input":
            return
        text = event.value.strip()
        if not text or self._processing:
            return

        # ── Slash-command handling ──
        if text.startswith("/"):
            self._handle_slash_command(text)
            return

        targets, clean_text = self._resolve_targets(text)
        if not clean_text:
            return

        room = self._current_room

        self._append_msg(room, "You", text)
        self._mount_bubble("You", text)

        if not targets and self._is_room(self._current_room):
            room_agent_names = self._room_agents(self._current_room)
            room_agent_defs = [
                d
                for d in self._agent_defs
                if d.get("name", d.get("role", "unnamed")) in room_agent_names
            ]
            engagement = self._room_engagement(self._current_room)

            pending_agent = self._find_agent_with_pending_suggestion()
            if pending_agent and pending_agent in room_agent_names:
                targets = [pending_agent]
            elif engagement == "organic":
                scored = await self._score_relevance_llm(clean_text, room_agent_defs)
                if scored is None:
                    scored = self._score_relevance(clean_text, room_agent_defs)
                if scored:
                    top_score = scored[0][1]
                    best = [scored[0][0]]
                    if (
                        len(scored) > 1
                        and scored[1][1] >= top_score * self._RELEVANCE_TIE_THRESHOLD
                    ):
                        best.append(scored[1][0])
                    targets = [d.get("name", d.get("role", "unnamed")) for d in best]
                else:
                    targets = [self._last_active_agent or room_agent_names[0]]
            elif len(room_agent_names) == 1:
                targets = [room_agent_names[0]]
            else:
                first = room_agent_names[0] if room_agent_names else "agent"
                self._mount_sys(
                    f"Tip: use @agent_name to direct your message, e.g. @{first}"
                )
                return

        self._processing = True

        if len(targets) == 1:
            thinking = ThinkingIndicator(targets[0])
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            near_bottom = self._is_near_bottom(scroll)
            scroll.mount(thinking)
            if near_bottom:
                scroll.scroll_end(animate=False)
            asyncio.ensure_future(self._process(targets[0], clean_text, thinking, room))
        else:
            asyncio.ensure_future(self._process_multi(targets, clean_text, room))

    # ── Organic mode relevance check (GAP-28) ──

    _RELEVANCE_LLM_MODEL: str = "anthropic/claude-haiku-4-5-20251001"

    async def _score_relevance_llm(
        self, message: str, agents: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], int]] | None:
        """Score agents by relevance using a cheap LLM (Haiku-tier).

        Returns scored list like _score_relevance(), or None on failure
        so the caller can fall back to the heuristic.
        """
        if not agents:
            return None
        try:
            from crewai.llm import LLM
        except Exception:
            return None

        agent_descriptions = "\n".join(
            f"- {d.get('name', d.get('role', 'unnamed'))}: "
            f"role={d.get('role', '')}, goal={d.get('goal', '')}"
            for d in agents
        )
        prompt = (
            "Given the user message and the list of available agents, "
            "return ONLY a JSON array of agent names that should respond, "
            "ordered by relevance (most relevant first). "
            "Include an agent only if the message is clearly relevant to its role/goal. "
            "Return an empty array if no agent is relevant.\n\n"
            f"Agents:\n{agent_descriptions}\n\n"
            f"User message: {message}\n\n"
            "Response (JSON array only):"
        )
        try:
            llm = LLM(model=self._RELEVANCE_LLM_MODEL)
            result = await asyncio.to_thread(
                llm.call, [{"role": "user", "content": prompt}]
            )
            text = str(result).strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )
            names = json.loads(text)
            if not isinstance(names, list):
                return None

            name_to_def = {d.get("name", d.get("role", "unnamed")): d for d in agents}
            scored: list[tuple[dict[str, Any], int]] = []
            for rank, name in enumerate(names):
                if name in name_to_def:
                    scored.append((name_to_def[name], len(names) - rank))
            return scored if scored else None
        except Exception:
            return None

    @staticmethod
    def _stem_words(words: set[str]) -> set[str]:
        """Simple suffix-stripping stemmer (GAP-108).

        Produces a superset: the original word plus a stemmed variant
        when a common English suffix is found.
        """
        stems: set[str] = set()
        for w in words:
            stems.add(w)
            if w.endswith("ing") and len(w) > 4:
                stems.add(w[:-3])
            elif w.endswith("ed") and len(w) > 3:
                stems.add(w[:-2])
            elif w.endswith("s") and len(w) > 2:
                stems.add(w[:-1])
        return stems

    _STOP_WORDS: set[str] = {
        "the",
        "a",
        "an",
        "is",
        "to",
        "and",
        "or",
        "of",
        "in",
        "it",
        "on",
        "for",
        "i",
        "my",
        "me",
        "can",
        "you",
        "do",
        "what",
        "how",
        "please",
        "help",
        "this",
        "that",
        "with",
        "are",
        "be",
        "was",
        "were",
        "has",
        "have",
        "had",
        "will",
        "would",
        "could",
        "should",
        "about",
        "just",
        "not",
        "but",
        "if",
        "they",
        "them",
        "their",
        "there",
        "here",
    }

    _RELEVANCE_TIE_THRESHOLD: float = 0.8

    def _score_relevance(
        self, message: str, agents: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], int]]:
        """Score agents by relevance to the message.

        Returns (agent_def, score) tuples sorted by score descending.
        Score = count of overlapping stems between the message and the
        agent's role, goal, and backstory fields.
        """
        msg_words = set(message.lower().split()) - self._STOP_WORDS
        msg_stems = self._stem_words(msg_words)

        scored: list[tuple[dict[str, Any], int]] = []
        for agent in agents:
            agent_text = " ".join(
                [
                    agent.get("role", ""),
                    agent.get("goal", ""),
                    agent.get("backstory", ""),
                ]
            ).lower()
            agent_words = set(agent_text.split()) - self._STOP_WORDS
            agent_stems = self._stem_words(agent_words)

            overlap = len(agent_stems & msg_stems)
            if overlap > 0:
                scored.append((agent, overlap))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    # ── Slash-command routing ──

    def _handle_slash_command(self, text: str) -> None:
        """Route /commands to their handlers."""
        parts = text.split(None, 2)
        cmd = parts[0].lower()

        if cmd == "/memory":
            self._handle_memory_command(parts)
        elif cmd == "/tasks":
            self._handle_tasks_command(parts)
        elif cmd == "/skills":
            self._handle_skills_command()
        else:
            self._mount_sys(f"Unknown command: {cmd}")

    def _handle_status_update(self, source: Any, event: Any) -> None:
        """Update the active ThinkingIndicator with structured progress."""
        state = getattr(event, "state", "")
        detail = getattr(event, "detail", None)
        input_tokens = getattr(event, "input_tokens", 0)
        output_tokens = getattr(event, "output_tokens", 0)

        try:
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            for child in reversed(scroll.children):
                if isinstance(child, ThinkingIndicator):
                    child.update_status(state, detail, input_tokens, output_tokens)
                    break
        except Exception:
            pass

    def _handle_skills_command(self) -> None:
        """List active skills for the current agent."""
        agent = None
        if not self._is_room(self._current_room):
            agent = self._get_or_create_agent(self._current_room)
        elif self._last_active_agent:
            agent = self._get_or_create_agent(self._last_active_agent)

        if agent is None:
            self._mount_sys("No agent selected.")
            return

        sb = getattr(agent, "_skill_builder", None)
        active = sb.get_active_skills() if sb else []

        if not active:
            self._mount_sys("No active skills for this agent.")
            return

        lines = [f"[bold]Active Skills[/] ({len(active)})"]
        for s in active:
            lines.append(f"  [{_CORAL}]{s.name}[/] — {s.description}")
        self._mount_sys("\n".join(lines))

    def _handle_tasks_command(self, parts: list[str]) -> None:
        """Show or cancel scheduled tasks."""
        try:
            from crewai.new_agent.scheduler import TaskScheduler

            scheduler = TaskScheduler()
        except Exception:
            self._mount_sys("Scheduler not available.")
            return

        if len(parts) > 1 and parts[1] == "cancel" and len(parts) > 2:
            task_id = parts[2].strip()
            if scheduler.cancel(task_id):
                self._mount_sys(f"Task '{task_id}' cancelled.")
            else:
                self._mount_sys(f"No pending task with id '{task_id}'.")
            return

        show_all = len(parts) > 1 and parts[1] == "all"
        tasks = scheduler.list_tasks(include_done=show_all)
        if not tasks:
            self._mount_sys(
                "No scheduled tasks." if not show_all else "No tasks found."
            )
            return

        lines: list[str] = [f"[bold]Scheduled Tasks[/] ({len(tasks)})"]
        for t in tasks:
            status_icon = {
                "pending": "◻",
                "running": "▶",
                "completed": "✓",
                "failed": "✗",
                "cancelled": "—",
            }.get(t.status, "?")
            agent = t.agent_name or "unknown"
            due = t.next_run_at[:16].replace("T", " ") if t.next_run_at else "—"
            line = (
                f"  {status_icon} [{_CORAL}]{t.id}[/] "
                f"[{_DIM}]{agent}[/] — {t.description[:60]}"
            )
            if t.status == "pending":
                line += f"  [dim]due {due}[/]"
            if t.schedule_type == "recurring":
                line += "  [dim](recurring)[/]"
            lines.append(line)
        lines.append(f"\n[{_DIM}]/tasks all — show completed  |  /tasks cancel <id>[/]")
        self._mount_sys("\n".join(lines))

    def _handle_memory_command(self, parts: list[str]) -> None:
        """Route /memory sub-commands."""
        if len(parts) == 1:
            # /memory — show recent memories for current agent
            self._show_memory_panel()
        elif parts[1] == "search" and len(parts) > 2:
            self._search_memory(parts[2])
        elif parts[1] == "clear":
            if len(parts) > 2 and parts[2].strip() == "confirm":
                self._clear_memory()
            else:
                self._mount_sys(
                    "Type [bold]/memory clear confirm[/] to delete all memories."
                )
        else:
            self._mount_sys(
                "Usage: /memory, /memory search <query>, /memory clear confirm"
            )

    def _get_focused_agent(self) -> Any:
        """Return the currently focused agent instance, or None."""
        if not self._is_room(self._current_room):
            return self._get_or_create_agent(self._current_room)
        if self._last_active_agent:
            return self._get_or_create_agent(self._last_active_agent)
        if self._agent_names:
            return self._get_or_create_agent(self._agent_names[0])
        return None

    def _format_memory_record(self, i: int, mem: object) -> list[str]:
        """Format a single memory record into Rich markup lines."""
        record = getattr(mem, "record", mem)
        content = getattr(record, "content", "") or str(mem)
        if len(content) > 150:
            content = content[:150] + "..."

        meta = getattr(record, "metadata", {}) or {}
        mem_type = meta.get("type", "raw")
        importance = getattr(record, "importance", "") or meta.get("importance", "")
        scope = getattr(record, "scope", "") or meta.get("scope", "")
        timestamp = getattr(record, "created_at", "")

        type_tag = (
            f"[bold cyan]{mem_type}[/]"
            if mem_type == "canonical"
            else f"[dim]{mem_type}[/]"
        )
        importance_tag = f" [yellow]★{importance}[/]" if importance else ""
        scope_tag = f" [{_DIM}]scope:{scope}[/]" if scope else ""
        time_tag = f" [{_DIM}]{timestamp}[/]" if timestamp else ""

        return [
            f"  {i}. {type_tag}{importance_tag}{scope_tag}{time_tag}",
            f"     {content}",
            "",
        ]

    def _show_memory_panel(self) -> None:
        """Show recent memories for the focused agent (GAP-92: rich formatting)."""
        agent = self._get_focused_agent()
        if agent is None:
            self._mount_sys("No agent selected.")
            return
        if not hasattr(agent, "_memory_instance") or not agent._memory_instance:
            self._mount_sys("No memories found for this agent.")
            return

        try:
            memories = agent._memory_instance.list_records(limit=10)
            if not memories:
                self._mount_sys("No memories stored yet.")
                return

            agent_name = getattr(agent, "role", "agent")
            lines = [f"[bold]Memory Inspector — {agent_name}[/]\n"]

            for i, mem in enumerate(memories, 1):
                lines.extend(self._format_memory_record(i, mem))

            lines.append(f"[{_DIM}]Use /memory search <query> to filter[/]")
            self._mount_sys("\n".join(lines))
        except Exception as e:
            self._mount_sys(f"Could not retrieve memories: {e}")

    def _search_memory(self, query: str) -> None:
        """Search agent memories by query (GAP-92: rich formatting)."""
        agent = self._get_focused_agent()
        if agent is None:
            self._mount_sys("No agent selected.")
            return
        if not hasattr(agent, "_memory_instance") or not agent._memory_instance:
            self._mount_sys("No memory available.")
            return

        try:
            results = agent._memory_instance.recall(query, limit=10, depth="shallow")
            if not results:
                self._mount_sys(f"No memories matching '{query}'")
                return

            agent_name = getattr(agent, "role", "agent")
            lines = [f"[bold]Memories matching '{query}' — {agent_name}[/]\n"]

            for i, mem in enumerate(results, 1):
                lines.extend(self._format_memory_record(i, mem))

            lines.append(f"[{_DIM}]Use /memory search <query> to refine[/]")
            self._mount_sys("\n".join(lines))
        except Exception as e:
            self._mount_sys(f"Memory search failed: {e}")

    def _clear_memory(self) -> None:
        """Clear all memories for the focused agent."""
        agent = self._get_focused_agent()
        if agent is None:
            self._mount_sys("No agent selected.")
            return
        if not hasattr(agent, "_memory_instance") or not agent._memory_instance:
            self._mount_sys("No memory to clear.")
            return

        try:
            agent._memory_instance.reset()
            agent_name = getattr(agent, "role", "agent")
            self._mount_sys(f"All memories cleared for {agent_name}.")
        except Exception as e:
            self._mount_sys(f"Failed to clear memories: {e}")

    # ── Message routing ──

    def _resolve_targets(self, text: str) -> tuple[list[str], str]:
        """Parse all @mentions in the message.

        Returns ``([agent_names], clean_text)``.
        In a room, at least one @mention is required for tagged mode —
        untagged messages return ``([], text)`` so the caller can handle routing.
        In a DM (agent name), messages always route to that agent.
        """
        found: list[str] = []
        clean = text
        for name in self._agent_names:
            pattern = re.compile(r"@" + re.escape(name) + r"\b", re.IGNORECASE)
            if pattern.search(clean):
                found.append(name)
                clean = pattern.sub("", clean).strip()
        if found:
            return found, clean
        if not self._is_room(self._current_room):
            return [self._current_room], text
        return [], text

    async def _process_multi(
        self,
        targets: list[str],
        text: str,
        room: str,
    ) -> None:
        """Process a message directed at multiple agents in parallel."""
        # Build room context once (shared snapshot before any replies)
        room_context: str | None = None
        if self._is_room(room):
            ctx = self._build_room_context(room)
            if ctx:
                room_context = (
                    "[Conversation so far]\n"
                    f"{ctx}\n\n"
                    "[Your turn — respond to the latest message]\n"
                    f"{text}"
                )

        # Mount a thinking indicator per agent
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        near_bottom = self._is_near_bottom(scroll)
        indicators: dict[str, ThinkingIndicator] = {}
        for target in targets:
            ind = ThinkingIndicator(target)
            indicators[target] = ind
            scroll.mount(ind)
        if near_bottom:
            scroll.scroll_end(animate=False)

        async def _call_agent(target: str) -> tuple[str, Any, Exception | None]:
            try:
                agent = await asyncio.to_thread(self._get_or_create_agent, target)
                if agent is None:
                    error_detail = getattr(self, "_last_agent_error", "")
                    detail = f": {error_detail}" if error_detail else ""
                    return (
                        target,
                        None,
                        ValueError(f"Could not load '{target}'{detail}"),
                    )
                msg = room_context if room_context else text
                resp = await asyncio.wait_for(
                    asyncio.to_thread(agent.message, msg),
                    timeout=300.0,
                )
                return target, resp, None
            except asyncio.TimeoutError:
                return (
                    target,
                    None,
                    TimeoutError(f"Agent '{target}' timed out after 5 minutes"),
                )
            except Exception as exc:
                return target, None, exc

        results = await asyncio.gather(*[_call_agent(t) for t in targets])

        for target, response, error in results:
            await self._safe_remove(indicators.get(target))  # type: ignore[arg-type]
            if error or response is None:
                msg = (
                    f"Error from {_safe_render(target)}: {_safe_render(str(error))}"
                    if error
                    else f"Could not load agent '{_safe_render(target)}'."
                )
                self._append_msg(room, "system", msg)
                if self._current_room == room:
                    self._mount_sys(msg)
                continue

            meta_parts: list[str] = []
            if response.input_tokens or response.output_tokens:
                meta_parts.append(
                    f"↑ {response.input_tokens or 0:,}  "
                    f"↓ {response.output_tokens or 0:,} tokens"
                )
            if response.response_time_ms:
                meta_parts.append(f"{response.response_time_ms / 1000:.1f}s")
            metadata = " · ".join(meta_parts)

            self._last_active_agent = target
            self._append_msg(room, target, response.content, metadata)
            if self._current_room == room:
                self._mount_bubble(target, response.content, metadata)

        self._processing = False
        self._save_history_to_disk()

    def _build_room_context(self, room: str, limit: int = 20) -> str:
        """Build a conversation transcript from the room history.

        Returned as a multi-line string the target agent can use to
        understand what was said before it was tagged.
        """
        history = self._chat_histories.get(room, [])
        # Only include user and agent messages (skip system)
        relevant = [
            (sender, content) for sender, content, _ in history if sender != "system"
        ]
        if not relevant:
            return ""
        recent = relevant[-limit:]
        lines: list[str] = []
        for sender, content in recent:
            prefix = "User" if sender == "You" else sender
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    async def _process(
        self,
        target: str | None,
        text: str,
        thinking: ThinkingIndicator,
        room: str,
    ) -> None:
        try:
            if target is None:
                await self._safe_remove(thinking)
                self._append_msg(room, "system", "No agent available.")
                if self._current_room == room:
                    self._mount_sys("No agent available.")
                return

            agent = await asyncio.wait_for(
                asyncio.to_thread(self._get_or_create_agent, target),
                timeout=60.0,
            )
            if agent is None:
                await self._safe_remove(thinking)
                error_detail = getattr(self, "_last_agent_error", "")
                if error_detail:
                    msg = f"Could not load agent '{_safe_render(target)}': {_safe_render(error_detail)}"
                else:
                    msg = f"Could not load agent '{_safe_render(target)}'."
                self._append_msg(room, "system", msg)
                if self._current_room == room:
                    self._mount_sys(msg)
                return

            message_text = text
            if self._is_room(room):
                ctx = self._build_room_context(room)
                if ctx:
                    message_text = (
                        "[Conversation so far]\n"
                        f"{ctx}\n\n"
                        "[Your turn — respond to the latest message]\n"
                        f"{text}"
                    )

            self._last_active_agent = target

            # Stream response token-by-token
            scroll = self.query_one("#chat-scroll", VerticalScroll)
            bubble: ChatBubble | None = None
            accumulated = ""
            stream_start = time.monotonic()
            stream_chars = 0

            def _stream_markup(
                text: str, final: bool = False, metadata: str = ""
            ) -> str:
                rendered = _safe_render(text)
                mk = f"[bold {_CORAL}]{target}[/]\n{rendered}"
                if final:
                    if metadata:
                        mk += f"\n\n[{_DIM}]{_safe_render(metadata)}[/]"
                else:
                    cursor = f"[{_CORAL}]▎[/]"
                    elapsed = time.monotonic() - stream_start
                    est_tokens = stream_chars // 4
                    progress = f"[{_DIM}]~{est_tokens:,} tokens · {elapsed:.1f}s[/]"
                    mk += f"{cursor}\n\n{progress}"
                return mk

            # Timeout-protected streaming: prevents UI freeze if LLM stalls
            stream = agent.stream(message_text)
            first_chunk = True
            while True:
                try:
                    timeout = 180.0 if first_chunk else 120.0
                    chunk = await asyncio.wait_for(anext(stream), timeout=timeout)
                    first_chunk = False
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError:
                    accumulated += "\n\n[Response timed out]"
                    break
                except Exception:
                    break

                accumulated += chunk
                stream_chars += len(chunk)

                if bubble is None and self._current_room == room:
                    try:
                        bubble = ChatBubble(
                            _stream_markup(accumulated), classes="agent-bubble"
                        )
                        scroll.mount(bubble, before=thinking)
                    except Exception:
                        bubble = ChatBubble(
                            _safe_render(accumulated), classes="agent-bubble"
                        )
                        scroll.mount(bubble, before=thinking)
                    if self._is_near_bottom(scroll):
                        scroll.scroll_end(animate=False)
                elif bubble is not None:
                    try:
                        bubble.update(_stream_markup(accumulated))
                    except Exception:
                        bubble.update(_safe_render(accumulated))
                    if self._is_near_bottom(scroll):
                        scroll.scroll_end(animate=False)

            # Remove cursor, add final metadata
            await self._safe_remove(thinking)

            response = getattr(agent, "last_stream_result", None)
            meta_parts: list[str] = []
            if response:
                if getattr(response, "input_tokens", 0) or getattr(
                    response, "output_tokens", 0
                ):
                    meta_parts.append(f"~{response.output_tokens or 0:,} tokens")
                if getattr(response, "response_time_ms", 0):
                    meta_parts.append(f"{response.response_time_ms / 1000:.1f}s")
            metadata = " · ".join(meta_parts)

            if bubble is not None:
                try:
                    bubble.update(
                        _stream_markup(accumulated, final=True, metadata=metadata)
                    )
                except Exception:
                    bubble.update(_safe_render(accumulated))

            content = accumulated or (response.content if response else "")
            self._append_msg(room, target, content, metadata)

        except Exception as e:
            await self._safe_remove(thinking)
            msg = f"Error: {_safe_render(str(e))}"
            self._append_msg(room, "system", msg)
            if self._current_room == room:
                self._mount_sys(msg)
        finally:
            self._processing = False
            self._save_history_to_disk()

    async def _safe_remove(self, widget: Static) -> None:
        try:
            await widget.remove()
        except Exception:
            pass

    # ── Agent management ──

    def _get_or_create_agent(self, name: str) -> Any:
        if name in self._agent_instances:
            return self._agent_instances[name]

        defn = next(
            (d for d in self._agent_defs if d.get("name", d.get("role", "")) == name),
            None,
        )
        if defn is None:
            return None

        try:
            from crewai.new_agent.definition_parser import load_agent_from_definition

            clean = {k: v for k, v in defn.items() if not k.startswith("_")}
            agent = load_agent_from_definition(clean, agents_dir=self._agents_dir)
            if agent is not None:
                self._agent_instances[name] = agent
            return agent
        except Exception as exc:
            self._last_agent_error = str(exc)
            return None

    # ── Chat rendering ──

    def _is_near_bottom(self, scroll: VerticalScroll) -> bool:
        """True when the user is scrolled to (or near) the bottom."""
        if scroll.max_scroll_y == 0:
            return True
        return scroll.scroll_y >= scroll.max_scroll_y - 80

    def _mount_bubble(self, sender: str, content: str, metadata: str = "") -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        near_bottom = self._is_near_bottom(scroll)
        scroll.mount(self._make_bubble(sender, content, metadata))
        if near_bottom:
            scroll.scroll_end(animate=False)

    def _mount_sys(self, text: str) -> None:
        """Mount a system message. Accepts pre-formatted Rich markup."""
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        near_bottom = self._is_near_bottom(scroll)
        try:
            bubble = ChatBubble(f"[{_DIM}]{text}[/]", classes="system-bubble")
        except Exception:
            bubble = ChatBubble(_safe_render(text), classes="system-bubble")
        scroll.mount(bubble)
        if near_bottom:
            scroll.scroll_end(animate=False)

    def _highlight_mentions(self, escaped_text: str) -> str:
        """Highlight @agent_name mentions in pre-escaped text."""
        for name in self._agent_names:
            escaped_text = re.sub(
                r"@" + re.escape(name) + r"\b",
                f"[bold {_TEAL}]@{name}[/]",
                escaped_text,
                flags=re.IGNORECASE,
            )
        return escaped_text

    def _make_bubble(self, sender: str, content: str, metadata: str = "") -> ChatBubble:
        if sender == "You":
            rendered = self._highlight_mentions(_safe_render(content))
            markup = f"[bold #e8e8e8]You[/]\n{rendered}"
            return ChatBubble(markup, classes="user-bubble")
        if sender == "system":
            try:
                return ChatBubble(f"[{_DIM}]{content}[/]", classes="system-bubble")
            except Exception:
                return ChatBubble(_safe_render(content), classes="system-bubble")
        rendered = _safe_render(content)
        markup = f"[bold {_CORAL}]{sender}[/]\n{rendered}"
        if metadata:
            markup += f"\n\n[{_DIM}]{_safe_render(metadata)}[/]"
        return ChatBubble(markup, classes="agent-bubble")

    def _render_chat(self) -> None:
        scroll = self.query_one("#chat-scroll", VerticalScroll)
        scroll.remove_children()

        history = self._chat_histories.get(self._current_room, [])

        if not history:
            if self._is_room(self._current_room):
                room_info = self._rooms.get(self._current_room, {})
                display = room_info.get("name", "room").replace("-", " ").title()
                room_agents = self._room_agents(self._current_room)
                names = ", ".join(room_agents[:5])
                engagement = self._room_engagement(self._current_room)
                if engagement == "organic":
                    self._mount_sys(
                        f"Welcome to {display}. "
                        f"Just type — relevant agents will respond. "
                        f"Use @agent_name to direct a message. Available: {names}"
                    )
                else:
                    self._mount_sys(
                        f"Welcome to {display}. "
                        f"Use @agent_name to chat. Available: {names}"
                    )
            else:
                self._mount_sys(
                    f"Chat with {self._current_room}. Type a message to begin."
                )
            return

        for sender, content, metadata in history:
            scroll.mount(self._make_bubble(sender, content, metadata))
        scroll.scroll_end(animate=False)

    # ── History persistence ──

    def _append_msg(
        self, room: str, sender: str, content: str, metadata: str = ""
    ) -> None:
        if room not in self._chat_histories:
            self._chat_histories[room] = []
        self._chat_histories[room].append((sender, content, metadata))

    def _save_history_to_disk(self) -> None:
        hdir = _history_dir()
        hdir.mkdir(parents=True, exist_ok=True)
        for room, msgs in self._chat_histories.items():
            safe = room.replace("/", "_").replace("\\", "_")
            path = hdir / f"{safe}.json"
            data = [{"sender": s, "content": c, "metadata": m} for s, c, m in msgs]
            try:
                path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                pass

    def _load_history_from_disk(self) -> None:
        hdir = _history_dir()
        if not hdir.exists():
            return
        for path in hdir.glob("*.json"):
            room = path.stem
            # Migrate old __common__ history to new __room__common key
            if room == "__common__":
                room = self._room_key("common")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                self._chat_histories[room] = [
                    (d["sender"], d["content"], d.get("metadata", "")) for d in data
                ]
            except Exception:
                pass

    # ── Sidebar: Memory tab ──

    def _launch_memory_browser(self) -> None:
        """Suspend this TUI and launch the memory browser as a subprocess."""
        import subprocess

        with self.suspend():
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from crewai_cli.memory_tui import MemoryTUI; MemoryTUI().run()",
                ],
            )

    def _find_agent_with_pending_suggestion(self) -> str | None:
        """Return the name of the first loaded agent that has a pending skill or knowledge suggestion."""
        for name, agent in self._agent_instances.items():
            sb = getattr(agent, "_skill_builder", None)
            if sb and sb.pending_suggestions:
                return name
            kd = getattr(agent, "_knowledge_discovery", None)
            if kd and getattr(kd, "pending_suggestions", None):
                return name
        return None

    def _get_focused_agent_name(self) -> str | None:
        """Return the agent name for the current room (DM only)."""
        if self._is_room(self._current_room):
            return self._last_active_agent
        if self._current_room in self._agent_names:
            return self._current_room
        return None

    def on_tabbed_content_tab_activated(
        self, event: TabbedContent.TabActivated
    ) -> None:
        pass

    # ── Sidebar buttons ──

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-memory":
            self._launch_memory_browser()
            return
        if event.button.id == "btn-new-room":
            self.push_screen(
                CreateRoomScreen(self._agent_names),
                callback=self._on_room_created,
            )
            return

    # ── Room creation ──

    def _on_room_created(self, result: dict[str, Any] | None) -> None:
        if result is None:
            return
        name = result["name"]
        key = self._room_key(name)
        if key in self._rooms:
            self._mount_sys(f"Room '{name}' already exists.")
            return

        self._rooms[key] = {
            "name": name,
            "agents": result["agents"],
            "engagement": result["engagement"],
        }

        # Add to sidebar
        room_list = self.query_one("#room-list", OptionList)
        display = name.replace("-", " ").title()
        engagement = result["engagement"]
        n_agents = len(result["agents"])
        room_list.add_option(f"◆ {display}  [{_DIM}]{engagement} · {n_agents}[/]")

        # Save to config.json
        self._save_room_to_config(name, result["agents"], result["engagement"])

        # Switch to the new room
        room_keys = list(self._rooms.keys())
        idx = room_keys.index(key)
        room_list.highlighted = idx
        self._current_room = key
        self._update_subtitle()
        self._update_placeholder()
        self._render_chat()
        self._mount_sys(f"Room '{display}' created with {n_agents} agent(s).")

    def _save_room_to_config(
        self, name: str, agents: list[str], engagement: str
    ) -> None:
        try:
            if self._config_path.exists():
                raw = self._config_path.read_text(encoding="utf-8")
                config = json.loads(_strip_jsonc(raw))
            else:
                config = {}
            rooms = config.setdefault("rooms", {})
            rooms[name] = {"agents": agents, "engagement": engagement}
            self._config_path.write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

    # ── Actions ──

    async def action_quit(self) -> None:
        """Graceful shutdown: stop scheduler, silence event bus, then exit."""
        self._mount_sys("Shutting down...")
        if self._scheduler:
            try:
                self._scheduler.stop()
            except Exception:
                pass
        try:
            from crewai.events.event_bus import crewai_event_bus

            crewai_event_bus.shutdown(wait=False)
        except Exception:
            pass
        self.exit()

    def action_clear_chat(self) -> None:
        self._chat_histories[self._current_room] = []
        self._render_chat()
        self._save_history_to_disk()


def _load_dotenv(base: Path) -> None:
    """Load .env file into os.environ if it exists."""
    try:
        for key, value in load_env_vars(base).items():
            if key not in os.environ:
                os.environ[key] = value
    except Exception:
        pass


def run_agent_tui(agents_dir: Path | None = None) -> None:
    """Launch the agent TUI."""
    base = Path.cwd()
    if agents_dir is None:
        agents_dir = base / "agents"

    if not agents_dir.is_dir():
        print(f"No agents/ directory found at {agents_dir}", file=sys.stderr)
        print("Create agents first: crewai create agent <name>", file=sys.stderr)
        raise SystemExit(1)

    files = list(agents_dir.glob("*.json")) + list(agents_dir.glob("*.jsonc"))
    if not files:
        print("No agent definitions found in agents/", file=sys.stderr)
        print("Create agents first: crewai create agent <name>", file=sys.stderr)
        raise SystemExit(1)

    _load_dotenv(base)
    config = _load_config(base)
    app = AgentTUI(agents_dir=agents_dir, config=config)
    app.run()
