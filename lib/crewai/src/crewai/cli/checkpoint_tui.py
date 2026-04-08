"""Textual TUI for browsing checkpoint files."""

from __future__ import annotations

from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, OptionList, Static
from textual.widgets.option_list import Option

from crewai.cli.checkpoint_cli import (
    _entity_summary,
    _format_size,
    _is_sqlite,
    _list_json,
    _list_sqlite,
)


_PRIMARY = "#eb6658"
_SECONDARY = "#1F7982"
_TERTIARY = "#ffffff"
_DIM = "#888888"
_BG_DARK = "#0d1117"
_BG_PANEL = "#161b22"


def _load_entries(location: str) -> list[dict[str, Any]]:
    if _is_sqlite(location):
        return _list_sqlite(location)
    return _list_json(location)


def _format_list_label(entry: dict[str, Any]) -> str:
    """Format a checkpoint entry for the list panel."""
    name = entry.get("name", "")
    ts = entry.get("ts") or ""
    trigger = entry.get("trigger") or ""
    summary = _entity_summary(entry.get("entities", []))

    line1 = f"[bold]{name}[/]"
    parts = []
    if ts:
        parts.append(f"[dim]{ts}[/]")
    if "size" in entry:
        parts.append(f"[dim]{_format_size(entry['size'])}[/]")
    if trigger:
        parts.append(f"[{_PRIMARY}]{trigger}[/]")
    line2 = "  ".join(parts)
    line3 = f"  [{_DIM}]{summary}[/]"

    return f"{line1}\n{line2}\n{line3}"


def _format_detail(entry: dict[str, Any]) -> str:
    """Format checkpoint details for the right panel."""
    lines: list[str] = []

    # Header
    name = entry.get("name", "")
    lines.append(f"[bold {_PRIMARY}]{name}[/]")
    lines.append(f"[{_DIM}]{'─' * 50}[/]")
    lines.append("")

    # Metadata table
    ts = entry.get("ts") or "unknown"
    trigger = entry.get("trigger") or ""
    lines.append(f"  [bold]Time[/]       {ts}")
    if "size" in entry:
        lines.append(f"  [bold]Size[/]       {_format_size(entry['size'])}")
    lines.append(f"  [bold]Events[/]     {entry.get('event_count', 0)}")
    if trigger:
        lines.append(f"  [bold]Trigger[/]    [{_PRIMARY}]{trigger}[/]")
    if "path" in entry:
        lines.append(f"  [bold]Path[/]       [{_DIM}]{entry['path']}[/]")
    if "db" in entry:
        lines.append(f"  [bold]Database[/]   [{_DIM}]{entry['db']}[/]")

    # Entities
    for ent in entry.get("entities", []):
        eid = str(ent.get("id", ""))[:8]
        etype = ent.get("type", "unknown")
        ename = ent.get("name", "unnamed")

        lines.append("")
        lines.append(f"  [{_DIM}]{'─' * 50}[/]")
        lines.append(f"  [bold {_SECONDARY}]{etype}[/]: {ename} [{_DIM}]{eid}[/]")

        tasks = ent.get("tasks")
        if isinstance(tasks, list):
            completed = ent.get("tasks_completed", 0)
            total = ent.get("tasks_total", 0)
            pct = int(completed / total * 100) if total else 0
            bar_len = 20
            filled = int(bar_len * completed / total) if total else 0
            bar = f"[{_PRIMARY}]{'█' * filled}[/][{_DIM}]{'░' * (bar_len - filled)}[/]"

            lines.append(f"  {bar}  {completed}/{total} tasks ({pct}%)")
            lines.append("")

            for i, task in enumerate(tasks):
                if task.get("completed"):
                    icon = "[green]✓[/]"
                else:
                    icon = "[yellow]○[/]"
                desc = str(task.get("description", ""))
                if len(desc) > 55:
                    desc = desc[:52] + "..."
                lines.append(f"    {icon}  {i + 1}. {desc}")

    return "\n".join(lines)


class ConfirmResumeScreen(ModalScreen[bool]):
    """Modal confirmation before resuming from a checkpoint."""

    CSS = f"""
    ConfirmResumeScreen {{
        align: center middle;
    }}
    #confirm-dialog {{
        width: 60;
        height: auto;
        padding: 1 2;
        background: {_BG_PANEL};
        border: round {_PRIMARY};
    }}
    #confirm-label {{
        width: 100%;
        content-align: center middle;
        margin-bottom: 1;
    }}
    #confirm-name {{
        width: 100%;
        content-align: center middle;
        color: {_PRIMARY};
        text-style: bold;
        margin-bottom: 1;
    }}
    #confirm-buttons {{
        width: 100%;
        height: 3;
        layout: horizontal;
        align: center middle;
    }}
    Button {{
        margin: 0 2;
        min-width: 12;
    }}
    """

    def __init__(self, checkpoint_name: str) -> None:
        super().__init__()
        self._checkpoint_name = checkpoint_name

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Static("Resume from this checkpoint?", id="confirm-label")
            yield Static(self._checkpoint_name, id="confirm-name")
            with Horizontal(id="confirm-buttons"):
                yield Button("Resume", variant="success", id="btn-yes")
                yield Button("Cancel", variant="default", id="btn-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-yes")

    def on_key(self, event: Any) -> None:
        if event.key == "y":
            self.dismiss(True)
        elif event.key in ("n", "escape"):
            self.dismiss(False)


class CheckpointTUI(App[str | None]):
    """TUI to browse and inspect checkpoints.

    Returns the checkpoint location string to resume from, or None if
    the user quit without selecting.
    """

    TITLE = "CrewAI Checkpoints"

    CSS = f"""
    Screen {{
        background: {_BG_DARK};
    }}
    Header {{
        background: {_PRIMARY};
        color: {_TERTIARY};
    }}
    Footer {{
        background: {_SECONDARY};
        color: {_TERTIARY};
    }}
    Footer > .footer-key--key {{
        background: {_PRIMARY};
        color: {_TERTIARY};
    }}
    Horizontal {{
        height: 1fr;
    }}
    #cp-list {{
        width: 38%;
        background: {_BG_PANEL};
        border: round {_SECONDARY};
        padding: 0 1;
        scrollbar-color: {_PRIMARY};
    }}
    #cp-list:focus {{
        border: round {_PRIMARY};
    }}
    #cp-list > .option-list--option-highlighted {{
        background: {_SECONDARY};
        color: {_TERTIARY};
        text-style: none;
    }}
    #cp-list > .option-list--option-highlighted * {{
        color: {_TERTIARY};
    }}
    #detail-container {{
        width: 62%;
        padding: 0 1;
    }}
    #detail {{
        height: 1fr;
        background: {_BG_PANEL};
        border: round {_SECONDARY};
        padding: 1 2;
        overflow-y: auto;
        scrollbar-color: {_PRIMARY};
    }}
    #detail:focus {{
        border: round {_PRIMARY};
    }}
    #status {{
        height: 1;
        padding: 0 2;
        color: {_DIM};
    }}
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
        ("j", "cursor_down", "Down"),
        ("k", "cursor_up", "Up"),
    ]

    def __init__(self, location: str = "./.checkpoints") -> None:
        super().__init__()
        self._location = location
        self._entries: list[dict[str, Any]] = []
        self._selected_idx: int = 0
        self._pending_location: str = ""

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            yield OptionList(id="cp-list")
            with Vertical(id="detail-container"):
                yield Static("", id="status")
                yield Static(
                    f"\n  [{_DIM}]Select a checkpoint from the list[/]",  # noqa: S608
                    id="detail",
                )
        yield Footer()

    async def on_mount(self) -> None:
        self.query_one("#cp-list", OptionList).border_title = "Checkpoints"
        self.query_one("#detail", Static).border_title = "Detail"
        self._refresh_list()

    def _refresh_list(self) -> None:
        self._entries = _load_entries(self._location)
        option_list = self.query_one("#cp-list", OptionList)
        option_list.clear_options()

        if not self._entries:
            self.query_one("#detail", Static).update(
                f"\n  [{_DIM}]No checkpoints in {self._location}[/]"
            )
            self.query_one("#status", Static).update("")
            self.sub_title = self._location
            return

        for entry in self._entries:
            option_list.add_option(Option(_format_list_label(entry)))

        count = len(self._entries)
        storage = "SQLite" if _is_sqlite(self._location) else "JSON"
        self.sub_title = f"{self._location}"
        self.query_one("#status", Static).update(f" {count} checkpoint(s) | {storage}")

    async def on_option_list_option_highlighted(
        self,
        event: OptionList.OptionHighlighted,
    ) -> None:
        idx = event.option_index
        if idx is None:
            return
        if idx < len(self._entries):
            self._selected_idx = idx
            entry = self._entries[idx]
            self.query_one("#detail", Static).update(_format_detail(entry))

    def action_cursor_down(self) -> None:
        self.query_one("#cp-list", OptionList).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#cp-list", OptionList).action_cursor_up()

    async def on_option_list_option_selected(
        self,
        event: OptionList.OptionSelected,
    ) -> None:
        idx = event.option_index
        if idx is None or idx >= len(self._entries):
            return
        entry = self._entries[idx]
        if "path" in entry:
            loc = entry["path"]
        elif _is_sqlite(self._location):
            loc = f"{self._location}#{entry['name']}"
        else:
            loc = entry.get("name", "")
        self._pending_location = loc
        name = entry.get("name", loc)
        self.push_screen(ConfirmResumeScreen(name), self._on_confirm)

    def _on_confirm(self, confirmed: bool | None) -> None:
        if confirmed:
            self.exit(self._pending_location)
        else:
            self._pending_location = ""

    def action_refresh(self) -> None:
        self._refresh_list()


async def _run_checkpoint_tui_async(location: str) -> None:
    """Async implementation of the checkpoint TUI flow."""
    import click

    app = CheckpointTUI(location=location)
    selected = await app.run_async()

    if selected is None:
        return

    click.echo(f"\nResuming from: {selected}\n")

    from crewai.crew import Crew

    crew = Crew.from_checkpoint(selected)
    result = await crew.akickoff()
    click.echo(f"\nResult: {getattr(result, 'raw', result)}")


def run_checkpoint_tui(location: str = "./.checkpoints") -> None:
    """Launch the checkpoint browser TUI."""
    import asyncio

    asyncio.run(_run_checkpoint_tui_async(location))
