"""Textual TUI for browsing checkpoint files."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Any, ClassVar, Literal

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Collapsible,
    Footer,
    Header,
    Input,
    Static,
    TabPane,
    TabbedContent,
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
_ACCENT = "#c9a227"
_SUCCESS = "#3fb950"
_PENDING = "#e3b341"

_ENTITY_ICONS: dict[str, str] = {
    "flow": "◆",
    "crew": "●",
    "agent": "◈",
    "unknown": "○",
}
_ENTITY_COLORS: dict[str, str] = {
    "flow": _ACCENT,
    "crew": _SECONDARY,
    "agent": _PRIMARY,
    "unknown": _DIM,
}


def _load_entries(location: str) -> list[dict[str, Any]]:
    if _is_sqlite(location):
        return _list_sqlite(location)
    return _list_json(location)


def _human_ts(ts: str) -> str:
    """Turn '2026-04-17 17:05:00' into a short relative label."""
    try:
        dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return ts
    now = datetime.now()
    delta = now.date() - dt.date()
    hour = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    time_str = f"{hour}:{dt.minute:02d}{ampm}"
    if delta.days == 0:
        return time_str
    if delta.days == 1:
        return f"yest {time_str}"
    if delta.days < 7:
        return f"{dt.strftime('%a').lower()} {time_str}"
    return f"{dt.strftime('%b')} {dt.day}"


def _short_id(name: str) -> str:
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


def _build_progress_bar(completed: int, total: int, width: int = 20) -> str:
    if total == 0:
        return f"[{_DIM}]{'░' * width}[/]  0/0"
    pct = int(completed / total * 100)
    filled = int(width * completed / total)
    color = _SUCCESS if completed == total else _PRIMARY
    bar = f"[{color}]{'█' * filled}[/][{_DIM}]{'░' * (width - filled)}[/]"
    return f"{bar}  {completed}/{total} ({pct}%)"


def _entity_icon(etype: str) -> str:
    icon = _ENTITY_ICONS.get(etype, _ENTITY_ICONS["unknown"])
    color = _ENTITY_COLORS.get(etype, _DIM)
    return f"[{color}]{icon}[/]"


_TuiResult = (
    tuple[
        str,
        str,
        dict[str, Any] | None,
        dict[int, str] | None,
        Literal["crew", "flow", "agent"],
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
        width: 40%;
        background: {_BG_PANEL};
        border: round {_SECONDARY};
        padding: 0 1;
        scrollbar-color: {_PRIMARY};
    }}
    #tree-panel:focus-within {{
        border: round {_PRIMARY};
    }}
    #detail-container {{
        width: 60%;
        height: 1fr;
    }}
    #status {{
        height: 1;
        padding: 0 2;
        color: {_DIM};
    }}
    #detail-tabs {{
        height: 1fr;
    }}
    TabbedContent > ContentSwitcher {{
        background: {_BG_PANEL};
        height: 1fr;
    }}
    TabPane {{
        padding: 0;
    }}
    Tabs {{
        background: {_BG_DARK};
    }}
    Tab {{
        background: {_BG_DARK};
        color: {_DIM};
        padding: 0 2;
    }}
    Tab.-active {{
        background: {_BG_PANEL};
        color: {_PRIMARY};
    }}
    Tab:hover {{
        color: {_TERTIARY};
    }}
    Underline > .underline--bar {{
        color: {_SECONDARY};
        background: {_BG_DARK};
    }}
    .tab-scroll {{
        background: {_BG_PANEL};
        height: 1fr;
        padding: 1 2;
        scrollbar-color: {_PRIMARY};
    }}
    .section-header {{
        padding: 0 0 0 1;
        margin: 1 0 0 0;
    }}
    .detail-line {{
        padding: 0 0 0 1;
    }}
    .task-label {{
        padding: 0 1;
    }}
    .task-output-editor {{
        height: auto;
        max-height: 10;
        margin: 0 1 1 3;
        border: round {_DIM};
    }}
    .task-output-editor:focus {{
        border: round {_PRIMARY};
    }}
    Collapsible {{
        background: {_BG_PANEL};
        padding: 0;
        margin: 0 0 1 1;
    }}
    CollapsibleTitle {{
        background: {_BG_DARK};
        color: {_TERTIARY};
        padding: 0 1;
    }}
    CollapsibleTitle:hover {{
        background: {_SECONDARY};
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
    .empty-state {{
        color: {_DIM};
        padding: 1;
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
        ("e", "resume", "Resume"),
        ("f", "fork", "Fork"),
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
            tree.show_root = False
            tree.guide_depth = 3
            yield tree
            with Vertical(id="detail-container"):
                yield Static("", id="status")
                with TabbedContent(id="detail-tabs"):
                    with TabPane("Overview", id="tab-overview"):
                        with VerticalScroll(classes="tab-scroll"):
                            yield Static(
                                f"[{_DIM}]Select a checkpoint from the tree[/]",  # noqa: S608
                                id="overview-empty",
                            )
                    with TabPane("Tasks", id="tab-tasks"):
                        with VerticalScroll(classes="tab-scroll"):
                            yield Static(
                                f"[{_DIM}]Select a checkpoint to view tasks[/]",
                                id="tasks-empty",
                            )
                    with TabPane("Inputs", id="tab-inputs"):
                        with VerticalScroll(classes="tab-scroll"):
                            yield Static(
                                f"[{_DIM}]Select a checkpoint to view inputs[/]",
                                id="inputs-empty",
                            )
        yield Footer()

    async def on_mount(self) -> None:
        self._refresh_tree()
        self.query_one("#tree-panel", Tree).root.expand()

    # ── Tree building ──────────────────────────────────────────────

    @staticmethod
    def _top_level_entity(entry: dict[str, Any]) -> tuple[str, str]:
        etype, ename = "unknown", ""
        for ent in entry.get("entities", []):
            t = ent.get("type", "unknown")
            if t == "flow":
                return "flow", ent.get("name") or ""
            if t == "crew" and etype != "crew":
                etype, ename = "crew", ent.get("name") or ""
        return etype, ename

    def _refresh_tree(self) -> None:
        self._entries = _load_entries(self._location)
        self._selected_entry = None

        tree = self.query_one("#tree-panel", Tree)
        tree.clear()

        if not self._entries:
            self.sub_title = self._location
            self.query_one("#status", Static).update("")
            return

        grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for entry in self._entries:
            key = self._top_level_entity(entry)
            branch = entry.get("branch", "main")
            grouped[key][branch].append(entry)

        def _make_label(e: dict[str, Any]) -> str:
            ts = e.get("ts") or ""
            trigger = e.get("trigger") or ""
            time_part = ts.split(" ")[-1] if " " in ts else ts

            total_c, total_t = 0, 0
            for ent in e.get("entities", []):
                c = ent.get("tasks_completed")
                t = ent.get("tasks_total")
                if c is not None and t is not None:
                    total_c += c
                    total_t += t

            parts: list[str] = []
            if time_part:
                parts.append(f"[{_DIM}]{time_part}[/]")
            if trigger:
                parts.append(f"[{_PRIMARY}]{trigger}[/]")
            if total_t:
                display_c = total_c
                if trigger == "task_started" and total_c < total_t:
                    display_c = total_c + 1
                color = _SUCCESS if total_c == total_t else _DIM
                parts.append(f"[{color}]{display_c}/{total_t}[/]")
            return "  ".join(parts) if parts else _short_id(e.get("name", ""))

        fork_parents: set[str] = set()
        for branches in grouped.values():
            for branch_name, entries in branches.items():
                if branch_name == "main" or not entries:
                    continue
                oldest = min(entries, key=lambda e: str(e.get("name", "")))
                first_parent = oldest.get("parent_id")
                if first_parent:
                    fork_parents.add(str(first_parent))

        node_by_name: dict[str, Any] = {}

        def _add_checkpoint(parent_node: Any, e: dict[str, Any]) -> None:
            cp_id = _entry_id(e)
            if cp_id in fork_parents:
                node = parent_node.add(
                    _make_label(e), data=e, expand=False, allow_expand=True
                )
            else:
                node = parent_node.add_leaf(_make_label(e), data=e)
            node_by_name[cp_id] = node

        type_order = {"flow": 0, "crew": 1}
        sorted_keys = sorted(
            grouped.keys(), key=lambda k: (type_order.get(k[0], 9), k[1])
        )

        for etype, ename in sorted_keys:
            branches = grouped[(etype, ename)]
            icon = _entity_icon(etype)
            color = _ENTITY_COLORS.get(etype, _DIM)
            total = sum(len(v) for v in branches.values())

            label_parts = [f"{icon} [bold {color}]{etype.upper()}[/]"]
            if ename:
                label_parts.append(f"[bold]{ename}[/]")
            label_parts.append(f"[{_DIM}]({total})[/]")
            all_entries = [e for bl in branches.values() for e in bl]
            timestamps = [str(e.get("ts", "")) for e in all_entries if e.get("ts")]
            if timestamps:
                latest = max(timestamps)
                label_parts.append(f"[{_DIM}]{_human_ts(latest)}[/]")
            entity_label = "  ".join(label_parts)
            entity_node = tree.root.add(entity_label, expand=True)

            if "main" in branches:
                for entry in reversed(branches["main"]):
                    _add_checkpoint(entity_node, entry)

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
                    attach_to: Any = entity_node
                    if first_parent:
                        attach_to = node_by_name.get(str(first_parent), entity_node)
                    branch_label = (
                        f"[bold {_SECONDARY}]{branch_name}[/]  "
                        f"[{_DIM}]({len(entries)})[/]"
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
                branch_node = entity_node.add(branch_label, expand=False)
                for entry in entries:
                    _add_checkpoint(branch_node, entry)

        count = len(self._entries)
        storage = "SQLite" if _is_sqlite(self._location) else "JSON"
        self.sub_title = self._location
        self.query_one("#status", Static).update(f" {count} checkpoint(s) | {storage}")

    # ── Detail panel ───────────────────────────────────────────────

    async def _clear_scroll(self, tab_id: str) -> VerticalScroll:
        tab = self.query_one(f"#{tab_id}", TabPane)
        scroll = tab.query_one(VerticalScroll)
        for child in list(scroll.children):
            await child.remove()
        return scroll

    async def _show_detail(self, entry: dict[str, Any]) -> None:
        self._selected_entry = entry

        await self._render_overview(entry)
        await self._render_tasks(entry)
        await self._render_inputs(entry.get("inputs", {}))

    async def _render_overview(self, entry: dict[str, Any]) -> None:
        scroll = await self._clear_scroll("tab-overview")

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

        await scroll.mount(Static("\n".join(header_lines)))

        for ent in entry.get("entities", []):
            etype = ent.get("type", "unknown")
            ename = ent.get("name", "unnamed")
            icon = _entity_icon(etype)
            color = _ENTITY_COLORS.get(etype, _DIM)

            eid = str(ent.get("id", ""))[:8]
            entity_title = (
                f"\n{icon} [bold {color}]{etype.upper()}[/]  [bold]{ename}[/]"
            )
            if eid:
                entity_title += f"  [{_DIM}]{eid}…[/]"
            await scroll.mount(Static(entity_title, classes="section-header"))
            await scroll.mount(Static(f"[{_DIM}]{'─' * 46}[/]", classes="detail-line"))

            if etype == "flow":
                methods = ent.get("completed_methods", [])
                if methods:
                    method_list = ", ".join(f"[{_SUCCESS}]{m}[/]" for m in methods)
                    await scroll.mount(
                        Static(
                            f"  [bold]Methods[/]  {method_list}",
                            classes="detail-line",
                        )
                    )
                flow_state = ent.get("flow_state")
                if isinstance(flow_state, dict) and flow_state:
                    state_parts: list[str] = []
                    for k, v in list(flow_state.items())[:5]:
                        sv = str(v)
                        if len(sv) > 40:
                            sv = sv[:37] + "..."
                        state_parts.append(f"[{_DIM}]{k}[/]={sv}")
                    await scroll.mount(
                        Static(
                            f"  [bold]State[/]    {', '.join(state_parts)}",
                            classes="detail-line",
                        )
                    )

            agents = ent.get("agents", [])
            if agents:
                agent_lines: list[Static] = []
                for ag in agents:
                    role = ag.get("role", "unnamed")
                    goal = ag.get("goal", "")
                    if len(goal) > 60:
                        goal = goal[:57] + "..."
                    agent_line = f"  {_entity_icon('agent')} [bold]{role}[/]"
                    if goal:
                        agent_line += f"\n    [{_DIM}]{goal}[/]"
                    agent_lines.append(Static(agent_line))

                collapsible = Collapsible(
                    *agent_lines,
                    title=f"Agents ({len(agents)})",
                    collapsed=len(agents) > 3,
                )
                await scroll.mount(collapsible)

    async def _render_tasks(self, entry: dict[str, Any]) -> None:
        scroll = await self._clear_scroll("tab-tasks")

        self._task_output_ids = []
        flat_task_idx = 0
        has_tasks = False

        for ent_idx, ent in enumerate(entry.get("entities", [])):
            etype = ent.get("type", "unknown")
            ename = ent.get("name", "unnamed")
            icon = _entity_icon(etype)
            color = _ENTITY_COLORS.get(etype, _DIM)

            tasks = ent.get("tasks", [])
            if not tasks:
                continue
            has_tasks = True

            completed = ent.get("tasks_completed", 0)
            total = ent.get("tasks_total", 0)

            await scroll.mount(
                Static(
                    f"{icon} [bold {color}]{ename}[/]  "
                    f"{_build_progress_bar(completed, total, width=16)}",
                    classes="section-header",
                )
            )

            for i, task in enumerate(tasks):
                desc = str(task.get("description", ""))
                if len(desc) > 50:
                    desc = desc[:47] + "..."
                agent_role = task.get("agent_role", "")

                if task.get("completed"):
                    status_icon = f"[{_SUCCESS}]✓[/]"
                    task_line = f"  {status_icon}  {i + 1}. {desc}"
                    if agent_role:
                        task_line += (
                            f"  [{_DIM}]→ {_entity_icon('agent')} {agent_role}[/]"
                        )
                    await scroll.mount(Static(task_line, classes="task-label"))
                    output_text = task.get("output", "")
                    editor_id = f"task-output-{ent_idx}-{i}"
                    await scroll.mount(
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
                    status_icon = f"[{_PENDING}]○[/]"
                    task_line = f"  {status_icon}  {i + 1}. {desc}"
                    if agent_role:
                        task_line += (
                            f"  [{_DIM}]→ {_entity_icon('agent')} {agent_role}[/]"
                        )
                    await scroll.mount(Static(task_line, classes="task-label"))
                flat_task_idx += 1

        if not has_tasks:
            await scroll.mount(Static(f"[{_DIM}]No tasks[/]", classes="empty-state"))

    async def _render_inputs(self, inputs: dict[str, Any]) -> None:
        scroll = await self._clear_scroll("tab-inputs")

        self._input_keys = []

        if not inputs:
            await scroll.mount(Static(f"[{_DIM}]No inputs[/]", classes="empty-state"))
            return

        for key, value in inputs.items():
            self._input_keys.append(key)
            row = Horizontal(classes="input-row")
            row.compose_add_child(Static(f"[bold]{key}[/]"))
            row.compose_add_child(
                Input(value=str(value), placeholder=key, id=f"input-{key}")
            )
            await scroll.mount(row)

    # ── Data collection ────────────────────────────────────────────

    def _collect_inputs(self) -> dict[str, Any] | None:
        if not self._input_keys:
            return None
        result: dict[str, Any] = {}
        for key in self._input_keys:
            widget = self.query_one(f"#input-{key}", Input)
            result[key] = widget.value
        return result

    def _collect_task_overrides(self) -> dict[int, str] | None:
        if not self._task_output_ids or self._selected_entry is None:
            return None
        overrides: dict[int, str] = {}
        for task_idx, editor_id, original in self._task_output_ids:
            editor = self.query_one(f"#{editor_id}", TextArea)
            if editor.text != original:
                overrides[task_idx] = editor.text
        return overrides or None

    def _detect_entity_type(
        self, entry: dict[str, Any]
    ) -> Literal["crew", "flow", "agent"]:
        for ent in entry.get("entities", []):
            if ent.get("type") == "flow":
                return "flow"
            if ent.get("type") == "agent":
                return "agent"
        return "crew"

    def _resolve_location(self, entry: dict[str, Any]) -> str:
        if "path" in entry:
            return str(entry["path"])
        if _is_sqlite(self._location):
            return f"{self._location}#{entry['name']}"
        return str(entry.get("name", ""))

    # ── Events ─────────────────────────────────────────────────────

    async def on_tree_node_highlighted(
        self, event: Tree.NodeHighlighted[dict[str, Any]]
    ) -> None:
        if event.node.data is not None:
            await self._show_detail(event.node.data)

    def _exit_with_action(self, action: str) -> None:
        if self._selected_entry is None:
            self.notify("No checkpoint selected", severity="warning")
            return
        inputs = self._collect_inputs()
        overrides = self._collect_task_overrides()
        loc = self._resolve_location(self._selected_entry)
        etype = self._detect_entity_type(self._selected_entry)
        name = self._selected_entry.get("name", "")[:30]
        self.notify(f"{action.title()}: {name}")
        self.exit((loc, action, inputs, overrides, etype))

    def action_resume(self) -> None:
        self._exit_with_action("resume")

    def action_fork(self) -> None:
        self._exit_with_action("fork")

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

    if entity_type == "agent":
        from crewai.agent import Agent

        if action == "fork":
            click.echo(f"\nForking agent from: {selected}\n")
            agent = Agent.fork(config)
        else:
            click.echo(f"\nResuming agent from: {selected}\n")
            agent = Agent.from_checkpoint(config)

        click.echo()
        result = await agent.akickoff(messages="Resume execution.")
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
