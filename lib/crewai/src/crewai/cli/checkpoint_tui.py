"""Textual TUI for browsing checkpoint files."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Static,
    Tree,
)

from crewai.cli.checkpoint_cli import (
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


def _short_id(name: str) -> str:
    """Shorten a checkpoint name for tree display."""
    if len(name) > 30:
        return name[:27] + "..."
    return name


def _build_entity_detail(ent: dict[str, Any]) -> str:
    """Build rich text for a single entity."""
    lines: list[str] = []
    eid = str(ent.get("id", ""))[:8]
    etype = ent.get("type", "unknown")
    ename = ent.get("name", "unnamed")
    lines.append(f"[bold {_SECONDARY}]{etype}[/]: {ename} [{_DIM}]{eid}[/]")

    tasks = ent.get("tasks")
    if isinstance(tasks, list):
        completed = ent.get("tasks_completed", 0)
        total = ent.get("tasks_total", 0)
        pct = int(completed / total * 100) if total else 0
        bar_len = 20
        filled = int(bar_len * completed / total) if total else 0
        bar = f"[{_PRIMARY}]{'█' * filled}[/][{_DIM}]{'░' * (bar_len - filled)}[/]"
        lines.append(f"{bar}  {completed}/{total} tasks ({pct}%)")
        lines.append("")
        for i, task in enumerate(tasks):
            icon = "[green]✓[/]" if task.get("completed") else "[yellow]○[/]"
            desc = str(task.get("description", ""))
            if len(desc) > 55:
                desc = desc[:52] + "..."
            lines.append(f"  {icon}  {i + 1}. {desc}")

    return "\n".join(lines)


class CheckpointTUI(App[tuple[str, str] | None]):
    """TUI to browse and inspect checkpoints.

    Returns ``(location, action)`` where action is ``"resume"`` or
    ``"fork"``, or ``None`` if the user quit without selecting.
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
    #main-layout {{
        height: 1fr;
    }}
    #tree-panel {{
        width: 45%;
        background: {_BG_PANEL};
        border: round {_SECONDARY};
        padding: 0 1;
        scrollbar-color: {_PRIMARY};
    }}
    #tree-panel:focus-within {{
        border: round {_PRIMARY};
    }}
    #detail-container {{
        width: 55%;
    }}
    #detail-scroll {{
        height: 1fr;
        background: {_BG_PANEL};
        border: round {_SECONDARY};
        padding: 1 2;
        scrollbar-color: {_PRIMARY};
    }}
    #detail-scroll:focus-within {{
        border: round {_PRIMARY};
    }}
    #detail-header {{
        margin-bottom: 1;
    }}
    #status {{
        height: 1;
        padding: 0 2;
        color: {_DIM};
    }}
    #action-buttons {{
        height: 3;
        align: center middle;
        padding: 0 1;
        display: none;
    }}
    #action-buttons.visible {{
        display: block;
    }}
    #action-buttons Button {{
        margin: 0 1;
        min-width: 14;
    }}
    #btn-resume {{
        background: {_SECONDARY};
        color: {_TERTIARY};
    }}
    #btn-resume:hover {{
        background: {_PRIMARY};
    }}
    #btn-fork {{
        background: {_PRIMARY};
        color: {_TERTIARY};
    }}
    #btn-fork:hover {{
        background: {_SECONDARY};
    }}
    Collapsible {{
        margin: 0;
        padding: 0;
    }}
    .entity-detail {{
        padding: 0 1;
    }}
    Tree {{
        background: {_BG_PANEL};
    }}
    Tree > .tree--cursor {{
        background: {_SECONDARY};
        color: {_TERTIARY};
    }}
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("q", "quit", "Quit"),
        ("r", "refresh", "Refresh"),
    ]

    def __init__(self, location: str = "./.checkpoints") -> None:
        super().__init__()
        self._location = location
        self._entries: list[dict[str, Any]] = []
        self._selected_entry: dict[str, Any] | None = None

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main-layout"):
            tree: Tree[dict[str, Any]] = Tree("Checkpoints", id="tree-panel")
            tree.show_root = True
            tree.guide_depth = 3
            yield tree
            with Vertical(id="detail-container"):
                yield Static("", id="status")
                with VerticalScroll(id="detail-scroll"):
                    yield Static(
                        f"[{_DIM}]Select a checkpoint from the tree[/]",  # noqa: S608
                        id="detail-header",
                    )
                with Horizontal(id="action-buttons"):
                    yield Button("Resume", id="btn-resume")
                    yield Button("Fork", id="btn-fork")
        yield Footer()

    async def on_mount(self) -> None:
        self._refresh_tree()
        self.query_one("#tree-panel", Tree).root.expand()

    def _refresh_tree(self) -> None:
        self._entries = _load_entries(self._location)
        self._selected_entry = None

        tree = self.query_one("#tree-panel", Tree)
        tree.clear()

        if not self._entries:
            self.query_one("#detail-header", Static).update(
                f"[{_DIM}]No checkpoints in {self._location}[/]"
            )
            self.query_one("#status", Static).update("")
            self.sub_title = self._location
            return

        # Group by branch
        branches: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for entry in self._entries:
            branch = entry.get("branch", "main")
            branches[branch].append(entry)

        # Index checkpoint names to tree nodes so forks can attach
        node_by_name: dict[str, Any] = {}

        def _make_label(entry: dict[str, Any]) -> str:
            name = entry.get("name", "")
            ts = entry.get("ts") or ""
            trigger = entry.get("trigger") or ""
            parts = [f"[bold]{_short_id(name)}[/]"]
            if ts:
                time_part = ts.split(" ")[-1] if " " in ts else ts
                parts.append(f"[{_DIM}]{time_part}[/]")
            if trigger:
                parts.append(f"[{_PRIMARY}]{trigger}[/]")
            return "  ".join(parts)

        # Find which checkpoints are fork parents so they get expandable nodes
        fork_parents: set[str] = set()
        for branch_name, entries in branches.items():
            if branch_name == "main":
                continue
            first_parent = (
                entries[-1].get("parent_id") if entries else None
            )  # reversed later; -1 is oldest
            if first_parent:
                fork_parents.add(str(first_parent))

        def _add_checkpoint(parent_node: Any, entry: dict[str, Any]) -> None:
            """Add a checkpoint node — expandable only if a fork attaches to it."""
            name = entry.get("name", "")
            if name in fork_parents:
                node = parent_node.add(
                    _make_label(entry), data=entry, expand=False, allow_expand=True
                )
            else:
                node = parent_node.add_leaf(_make_label(entry), data=entry)
            node_by_name[name] = node

        # Build main branch directly under root (oldest to newest)
        if "main" in branches:
            for entry in reversed(branches["main"]):
                _add_checkpoint(tree.root, entry)

        # Build fork branches — sort so parent forks are built before child forks
        fork_branches = [
            (name, list(reversed(entries)))
            for name, entries in branches.items()
            if name != "main"
        ]
        # Process forks whose parent is already indexed first
        remaining = fork_branches
        max_passes = len(remaining) + 1
        while remaining and max_passes > 0:
            max_passes -= 1
            deferred = []
            for branch_name, entries in remaining:
                first_parent = entries[0].get("parent_id") if entries else None
                if first_parent and str(first_parent) not in node_by_name:
                    deferred.append((branch_name, entries))
                    continue
                attach_to = (
                    node_by_name.get(str(first_parent), tree.root)
                    if first_parent
                    else tree.root
                )
                branch_label = (
                    f"[bold {_SECONDARY}]{branch_name}[/]  [{_DIM}]({len(entries)})[/]"
                )
                branch_node = attach_to.add(branch_label, expand=False)
                for entry in entries:
                    _add_checkpoint(branch_node, entry)
            remaining = deferred

        count = len(self._entries)
        storage = "SQLite" if _is_sqlite(self._location) else "JSON"
        self.sub_title = self._location
        self.query_one("#status", Static).update(f" {count} checkpoint(s) | {storage}")

    def _show_detail(self, entry: dict[str, Any]) -> None:
        """Update the detail panel for a checkpoint entry."""
        self._selected_entry = entry
        self.query_one("#action-buttons").add_class("visible")

        detail_scroll = self.query_one("#detail-scroll", VerticalScroll)

        # Remove old collapsibles
        for widget in list(detail_scroll.query("Collapsible")):
            widget.remove()

        # Header
        name = entry.get("name", "")
        ts = entry.get("ts") or "unknown"
        trigger = entry.get("trigger") or ""
        branch = entry.get("branch", "main")
        parent_id = entry.get("parent_id")

        header_lines = [
            f"[bold {_PRIMARY}]{name}[/]",
            f"[{_DIM}]{'─' * 50}[/]",
            "",
            f"  [bold]Time[/]       {ts}",
        ]
        if "size" in entry:
            header_lines.append(f"  [bold]Size[/]       {_format_size(entry['size'])}")
        header_lines.append(f"  [bold]Events[/]     {entry.get('event_count', 0)}")
        if trigger:
            header_lines.append(f"  [bold]Trigger[/]    [{_PRIMARY}]{trigger}[/]")
        header_lines.append(f"  [bold]Branch[/]     [{_SECONDARY}]{branch}[/]")
        if parent_id:
            header_lines.append(f"  [bold]Parent[/]     [{_DIM}]{parent_id}[/]")
        if "path" in entry:
            header_lines.append(f"  [bold]Path[/]       [{_DIM}]{entry['path']}[/]")
        if "db" in entry:
            header_lines.append(f"  [bold]Database[/]   [{_DIM}]{entry['db']}[/]")

        self.query_one("#detail-header", Static).update("\n".join(header_lines))

        # Entity collapsibles
        for ent in entry.get("entities", []):
            etype = ent.get("type", "unknown")
            ename = ent.get("name", "unnamed")
            completed = ent.get("tasks_completed")
            total = ent.get("tasks_total")
            title = f"{etype}: {ename}"
            if completed is not None and total is not None:
                title += f" [{completed}/{total} tasks]"

            content = Static(_build_entity_detail(ent), classes="entity-detail")
            collapsible = Collapsible(content, title=title, collapsed=False)
            detail_scroll.mount(collapsible)

    def _resolve_location(self, entry: dict[str, Any]) -> str:
        """Get the restore location string for a checkpoint entry."""
        if "path" in entry:
            return str(entry["path"])
        if _is_sqlite(self._location):
            return f"{self._location}#{entry['name']}"
        return str(entry.get("name", ""))

    async def on_tree_node_highlighted(
        self, event: Tree.NodeHighlighted[dict[str, Any]]
    ) -> None:
        if event.node.data is not None:
            self._show_detail(event.node.data)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._selected_entry is None:
            return
        loc = self._resolve_location(self._selected_entry)
        if event.button.id == "btn-resume":
            self.exit((loc, "resume"))
        elif event.button.id == "btn-fork":
            self.exit((loc, "fork"))

    def action_refresh(self) -> None:
        self._refresh_tree()


async def _run_checkpoint_tui_async(location: str) -> None:
    """Async implementation of the checkpoint TUI flow."""
    import click

    app = CheckpointTUI(location=location)
    selection = await app.run_async()

    if selection is None:
        return

    selected, action = selection

    from crewai.crew import Crew
    from crewai.state.checkpoint_config import CheckpointConfig

    config = CheckpointConfig(restore_from=selected)

    if action == "fork":
        click.echo(f"\nForking from: {selected}\n")
        crew = Crew.fork(config)
    else:
        click.echo(f"\nResuming from: {selected}\n")
        crew = Crew.from_checkpoint(config)

    result = await crew.akickoff()
    click.echo(f"\nResult: {getattr(result, 'raw', result)}")


def run_checkpoint_tui(location: str = "./.checkpoints") -> None:
    """Launch the checkpoint browser TUI."""
    import asyncio

    asyncio.run(_run_checkpoint_tui_async(location))
