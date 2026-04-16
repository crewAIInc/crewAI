"""Textual TUI for browsing checkpoint files."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar, Literal

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Input,
    Static,
    TextArea,
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


def _entry_id(entry: dict[str, Any]) -> str:
    """Normalize an entry's name into its checkpoint ID.

    JSON filenames are ``{ts}_{uuid}_p-{parent}.json``; SQLite IDs
    are already ``{ts}_{uuid}``. This strips the JSON suffix so
    fork-parent lookups work in both providers.
    """
    name = str(entry.get("name", ""))
    if name.endswith(".json"):
        name = name[: -len(".json")]
    idx = name.find("_p-")
    if idx != -1:
        name = name[:idx]
    return name


def _build_entity_header(ent: dict[str, Any]) -> str:
    """Build rich text header for an entity (progress bar only)."""
    lines: list[str] = []
    tasks = ent.get("tasks")
    if isinstance(tasks, list):
        completed = ent.get("tasks_completed", 0)
        total = ent.get("tasks_total", 0)
        pct = int(completed / total * 100) if total else 0
        bar_len = 20
        filled = int(bar_len * completed / total) if total else 0
        bar = f"[{_PRIMARY}]{'█' * filled}[/][{_DIM}]{'░' * (bar_len - filled)}[/]"
        lines.append(f"{bar}  {completed}/{total} tasks ({pct}%)")
    return "\n".join(lines)


# Return type: (location, action, inputs, task_output_overrides, entity_type)
_TuiResult = (
    tuple[
        str,
        str,
        dict[str, Any] | None,
        dict[int, str] | None,
        Literal["crew", "flow"],
    ]
    | None
)


class CheckpointTUI(App[_TuiResult]):
    """TUI to browse and inspect checkpoints.

    Returns ``(location, action, inputs, task_overrides, entity_type)``
    where action is ``"resume"`` or ``"fork"``, inputs is a parsed dict
    or ``None``, and entity_type is ``"crew"`` or ``"flow"``;
    or ``None`` if the user quit without selecting.
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
        height: 1fr;
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
    #inputs-section {{
        display: none;
        height: auto;
        max-height: 8;
        padding: 0 1;
    }}
    #inputs-section.visible {{
        display: block;
    }}
    #inputs-label {{
        height: 1;
        color: {_DIM};
        padding: 0 1;
    }}
    .input-row {{
        height: 3;
        padding: 0 1;
    }}
    .input-row Static {{
        width: auto;
        min-width: 12;
        padding: 1 1 0 0;
        color: {_TERTIARY};
    }}
    .input-row Input {{
        width: 1fr;
    }}
    #no-inputs-label {{
        height: 1;
        color: {_DIM};
        padding: 0 1;
    }}
    #action-buttons {{
        height: 3;
        align: right middle;
        padding: 0 1;
        display: none;
    }}
    #action-buttons.visible {{
        display: block;
    }}
    #action-buttons Button {{
        margin: 0 0 0 1;
        min-width: 10;
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
    .entity-title {{
        padding: 1 1 0 1;
    }}
    .entity-detail {{
        padding: 0 1;
    }}
    .task-output-editor {{
        height: auto;
        max-height: 10;
        margin: 0 1 1 1;
        border: round {_DIM};
    }}
    .task-output-editor:focus {{
        border: round {_PRIMARY};
    }}
    .task-label {{
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
        self._input_keys: list[str] = []
        self._task_output_ids: list[tuple[int, str, str]] = []

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
                with Vertical(id="inputs-section"):
                    yield Static("Inputs", id="inputs-label")
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

        def _make_label(e: dict[str, Any]) -> str:
            name = e.get("name", "")
            ts = e.get("ts") or ""
            trigger = e.get("trigger") or ""
            parts = [f"[bold]{_short_id(name)}[/]"]
            if ts:
                time_part = ts.split(" ")[-1] if " " in ts else ts
                parts.append(f"[{_DIM}]{time_part}[/]")
            if trigger:
                parts.append(f"[{_PRIMARY}]{trigger}[/]")
            return "  ".join(parts)

        fork_parents: set[str] = set()
        for branch_name, entries in branches.items():
            if branch_name == "main" or not entries:
                continue
            oldest = min(entries, key=lambda e: str(e.get("name", "")))
            first_parent = oldest.get("parent_id")
            if first_parent:
                fork_parents.add(str(first_parent))

        def _add_checkpoint(parent_node: Any, e: dict[str, Any]) -> None:
            """Add a checkpoint node — expandable only if a fork attaches to it."""
            cp_id = _entry_id(e)
            if cp_id in fork_parents:
                node = parent_node.add(
                    _make_label(e), data=e, expand=False, allow_expand=True
                )
            else:
                node = parent_node.add_leaf(_make_label(e), data=e)
            node_by_name[cp_id] = node

        if "main" in branches:
            for entry in reversed(branches["main"]):
                _add_checkpoint(tree.root, entry)

        fork_branches = [
            (name, sorted(entries, key=lambda e: str(e.get("name", ""))))
            for name, entries in branches.items()
            if name != "main"
        ]
        remaining = fork_branches
        max_passes = len(remaining) + 1
        while remaining and max_passes > 0:
            max_passes -= 1
            deferred = []
            made_progress = False
            for branch_name, entries in remaining:
                first_parent = entries[0].get("parent_id") if entries else None
                if first_parent and str(first_parent) not in node_by_name:
                    deferred.append((branch_name, entries))
                    continue
                attach_to: Any = tree.root
                if first_parent:
                    attach_to = node_by_name.get(str(first_parent), tree.root)
                branch_label = (
                    f"[bold {_SECONDARY}]{branch_name}[/]  [{_DIM}]({len(entries)})[/]"
                )
                branch_node = attach_to.add(branch_label, expand=False)
                for entry in entries:
                    _add_checkpoint(branch_node, entry)
                made_progress = True
            remaining = deferred
            if not made_progress:
                break

        for branch_name, entries in remaining:
            branch_label = (
                f"[bold {_SECONDARY}]{branch_name}[/]  "
                f"[{_DIM}]({len(entries)})[/]  [{_DIM}](orphaned)[/]"
            )
            branch_node = tree.root.add(branch_label, expand=False)
            for entry in entries:
                _add_checkpoint(branch_node, entry)

        count = len(self._entries)
        storage = "SQLite" if _is_sqlite(self._location) else "JSON"
        self.sub_title = self._location
        self.query_one("#status", Static).update(f" {count} checkpoint(s) | {storage}")

    async def _show_detail(self, entry: dict[str, Any]) -> None:
        """Update the detail panel for a checkpoint entry."""
        self._selected_entry = entry
        self.query_one("#action-buttons").add_class("visible")

        detail_scroll = self.query_one("#detail-scroll", VerticalScroll)

        # Remove all dynamic children except the header — await so IDs are freed
        to_remove = [c for c in detail_scroll.children if c.id != "detail-header"]
        for child in to_remove:
            await child.remove()

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

        # Entity details and editable task outputs — mounted flat for scrolling
        self._task_output_ids = []
        flat_task_idx = 0
        for ent_idx, ent in enumerate(entry.get("entities", [])):
            etype = ent.get("type", "unknown")
            ename = ent.get("name", "unnamed")
            completed = ent.get("tasks_completed")
            total = ent.get("tasks_total")
            entity_title = f"[bold {_SECONDARY}]{etype}: {ename}[/]"
            if completed is not None and total is not None:
                entity_title += f"  [{_DIM}]{completed}/{total} tasks[/]"
            await detail_scroll.mount(Static(entity_title, classes="entity-title"))
            await detail_scroll.mount(
                Static(_build_entity_header(ent), classes="entity-detail")
            )

            tasks = ent.get("tasks", [])
            for i, task in enumerate(tasks):
                desc = str(task.get("description", ""))
                if len(desc) > 55:
                    desc = desc[:52] + "..."
                if task.get("completed"):
                    icon = "[green]✓[/]"
                    await detail_scroll.mount(
                        Static(f"  {icon}  {i + 1}. {desc}", classes="task-label")
                    )
                    output_text = task.get("output", "")
                    editor_id = f"task-output-{ent_idx}-{i}"
                    await detail_scroll.mount(
                        TextArea(
                            str(output_text),
                            classes="task-output-editor",
                            id=editor_id,
                        )
                    )
                    self._task_output_ids.append(
                        (flat_task_idx, editor_id, str(output_text))
                    )
                else:
                    icon = "[yellow]○[/]"
                    await detail_scroll.mount(
                        Static(f"  {icon}  {i + 1}. {desc}", classes="task-label")
                    )
                flat_task_idx += 1

        # Build input fields
        await self._build_input_fields(entry.get("inputs", {}))

    async def _build_input_fields(self, inputs: dict[str, Any]) -> None:
        """Rebuild the inputs section with one field per input key."""
        section = self.query_one("#inputs-section")

        # Remove old dynamic children — await so IDs are freed
        for widget in list(section.query(".input-row, .no-inputs")):
            await widget.remove()

        self._input_keys = []

        if not inputs:
            await section.mount(Static(f"[{_DIM}]No inputs[/]", classes="no-inputs"))
            section.add_class("visible")
            return

        for key, value in inputs.items():
            self._input_keys.append(key)
            row = Horizontal(classes="input-row")
            row.compose_add_child(Static(f"[bold]{key}[/]"))
            row.compose_add_child(
                Input(value=str(value), placeholder=key, id=f"input-{key}")
            )
            await section.mount(row)

        section.add_class("visible")

    def _collect_inputs(self) -> dict[str, Any] | None:
        """Collect current values from input fields."""
        if not self._input_keys:
            return None
        result: dict[str, Any] = {}
        for key in self._input_keys:
            widget = self.query_one(f"#input-{key}", Input)
            result[key] = widget.value
        return result

    def _collect_task_overrides(self) -> dict[int, str] | None:
        """Collect edited task outputs. Returns only changed values."""
        if not self._task_output_ids or self._selected_entry is None:
            return None
        overrides: dict[int, str] = {}
        for task_idx, editor_id, original in self._task_output_ids:
            editor = self.query_one(f"#{editor_id}", TextArea)
            if editor.text != original:
                overrides[task_idx] = editor.text
        return overrides or None

    def _detect_entity_type(self, entry: dict[str, Any]) -> Literal["crew", "flow"]:
        """Infer the top-level entity type from checkpoint entities."""
        for ent in entry.get("entities", []):
            if ent.get("type") == "flow":
                return "flow"
        return "crew"

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
            await self._show_detail(event.node.data)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if self._selected_entry is None:
            return
        inputs = self._collect_inputs()
        overrides = self._collect_task_overrides()
        loc = self._resolve_location(self._selected_entry)
        etype = self._detect_entity_type(self._selected_entry)
        if event.button.id == "btn-resume":
            self.exit((loc, "resume", inputs, overrides, etype))
        elif event.button.id == "btn-fork":
            self.exit((loc, "fork", inputs, overrides, etype))

    def action_refresh(self) -> None:
        self._refresh_tree()


def _apply_task_overrides(crew: Any, task_overrides: dict[int, str]) -> None:
    """Apply task output overrides to a restored Crew and print modifications."""
    import click

    click.echo("Modifications:")
    overridden_agents: set[int] = set()
    for task_idx, new_output in task_overrides.items():
        if task_idx < len(crew.tasks) and crew.tasks[task_idx].output is not None:
            desc = crew.tasks[task_idx].description or f"Task {task_idx + 1}"
            if len(desc) > 60:
                desc = desc[:57] + "..."
            crew.tasks[task_idx].output.raw = new_output
            preview = new_output.replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:77] + "..."
            click.echo(f"  Task {task_idx + 1}: {desc}")
            click.echo(f"    -> {preview}")
            agent = crew.tasks[task_idx].agent
            if agent and agent.agent_executor:
                nth = sum(1 for t in crew.tasks[:task_idx] if t.agent is agent)
                messages = agent.agent_executor.messages
                system_positions = [
                    i for i, m in enumerate(messages) if m.get("role") == "system"
                ]
                if nth < len(system_positions):
                    seg_start = system_positions[nth]
                    seg_end = (
                        system_positions[nth + 1]
                        if nth + 1 < len(system_positions)
                        else len(messages)
                    )
                    for j in range(seg_end - 1, seg_start, -1):
                        if messages[j].get("role") == "assistant":
                            messages[j]["content"] = new_output
                            break
                overridden_agents.add(id(agent))

    earliest = min(task_overrides)
    for offset, subsequent in enumerate(crew.tasks[earliest + 1 :], start=earliest + 1):
        if subsequent.output and offset not in task_overrides:
            subsequent.output = None
        if subsequent.agent and subsequent.agent.agent_executor:
            subsequent.agent.agent_executor._resuming = False
            if id(subsequent.agent) not in overridden_agents:
                subsequent.agent.agent_executor.messages = []
    click.echo()


async def _run_checkpoint_tui_async(location: str) -> None:
    """Async implementation of the checkpoint TUI flow."""
    import click

    app = CheckpointTUI(location=location)
    selection = await app.run_async()

    if selection is None:
        return

    selected, action, inputs, task_overrides, entity_type = selection

    from crewai.state.checkpoint_config import CheckpointConfig

    config = CheckpointConfig(restore_from=selected)

    if entity_type == "flow":
        from crewai.events.event_bus import crewai_event_bus
        from crewai.flow.flow import Flow

        if action == "fork":
            click.echo(f"\nForking flow from: {selected}\n")
            flow = Flow.fork(config)
        else:
            click.echo(f"\nResuming flow from: {selected}\n")
            flow = Flow.from_checkpoint(config)

        if task_overrides:
            from crewai.crew import Crew as CrewCls

            state = crewai_event_bus._runtime_state
            if state is not None:
                flat_offset = 0
                for entity in state.root:
                    if not isinstance(entity, CrewCls) or not entity.tasks:
                        continue
                    n = len(entity.tasks)
                    local = {
                        idx - flat_offset: out
                        for idx, out in task_overrides.items()
                        if flat_offset <= idx < flat_offset + n
                    }
                    if local:
                        _apply_task_overrides(entity, local)
                    flat_offset += n

        if inputs:
            click.echo("Inputs:")
            for k, v in inputs.items():
                click.echo(f"  {k}: {v}")
            click.echo()

        result = await flow.kickoff_async(inputs=inputs)
        click.echo(f"\nResult: {getattr(result, 'raw', result)}")
        return

    from crewai.crew import Crew

    if action == "fork":
        click.echo(f"\nForking from: {selected}\n")
        crew = Crew.fork(config)
    else:
        click.echo(f"\nResuming from: {selected}\n")
        crew = Crew.from_checkpoint(config)

    if task_overrides:
        _apply_task_overrides(crew, task_overrides)

    if inputs:
        click.echo("Inputs:")
        for k, v in inputs.items():
            click.echo(f"  {k}: {v}")
        click.echo()

    result = await crew.akickoff(inputs=inputs)
    click.echo(f"\nResult: {getattr(result, 'raw', result)}")


def run_checkpoint_tui(location: str = "./.checkpoints") -> None:
    """Launch the checkpoint browser TUI."""
    import asyncio

    asyncio.run(_run_checkpoint_tui_async(location))
