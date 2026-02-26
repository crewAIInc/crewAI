"""Textual TUI for browsing and recalling unified memory."""

from __future__ import annotations

import asyncio
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Input, OptionList, Static, Tree


# -- CrewAI brand palette --
_PRIMARY = "#eb6658"  # coral
_SECONDARY = "#1F7982"  # teal
_TERTIARY = "#ffffff"  # white


def _format_scope_info(info: Any) -> str:
    """Format ScopeInfo with Rich markup."""
    return (
        f"[bold {_PRIMARY}]{info.path}[/]\n\n"
        f"[dim]Records:[/]     [bold]{info.record_count}[/]\n"
        f"[dim]Categories:[/]  {', '.join(info.categories) or 'none'}\n"
        f"[dim]Oldest:[/]      {info.oldest_record or '-'}\n"
        f"[dim]Newest:[/]      {info.newest_record or '-'}\n"
        f"[dim]Children:[/]    {', '.join(info.child_scopes) or 'none'}"
    )


class MemoryTUI(App[None]):
    """TUI to browse memory scopes and run recall queries."""

    TITLE = "CrewAI Memory"
    SUB_TITLE = "Browse scopes and recall memories"

    CSS = f"""
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
    #scope-tree {{
        width: 30%;
        padding: 1 2;
        background: {_SECONDARY} 8%;
        border-right: solid {_SECONDARY};
    }}
    #scope-tree:focus > .tree--cursor {{
        background: {_SECONDARY};
        color: {_TERTIARY};
    }}
    #scope-tree > .tree--guides {{
        color: {_SECONDARY} 50%;
    }}
    #scope-tree > .tree--guides-hover {{
        color: {_PRIMARY};
    }}
    #scope-tree > .tree--guides-selected {{
        color: {_SECONDARY};
    }}
    #right-panel {{
        width: 70%;
        padding: 0 1;
    }}
    #info-panel {{
        height: 2fr;
        padding: 1 2;
        overflow-y: auto;
        border: round {_SECONDARY};
    }}
    #info-panel:focus {{
        border: round {_PRIMARY};
    }}
    #info-panel LoadingIndicator {{
        color: {_PRIMARY};
    }}
    #entry-list {{
        height: 1fr;
        border: round {_SECONDARY};
        padding: 0 1;
        scrollbar-color: {_PRIMARY};
    }}
    #entry-list:focus {{
        border: round {_PRIMARY};
    }}
    #entry-list > .option-list--option-highlighted {{
        background: {_SECONDARY};
        color: {_TERTIARY};
    }}
    #recall-input {{
        margin: 0 1 1 1;
        border: tall {_SECONDARY};
    }}
    #recall-input:focus {{
        border: tall {_PRIMARY};
    }}
    """

    def __init__(
        self,
        storage_path: str | None = None,
        embedder_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._memory: Any = None
        self._init_error: str | None = None
        self._selected_scope: str = "/"
        self._entries: list[Any] = []
        self._view_mode: str = "list"  # "list" | "recall"
        self._recall_matches: list[Any] = []
        self._last_scope_info: Any = None
        self._custom_embedder = embedder_config is not None
        try:
            from crewai.memory.storage.lancedb_storage import LanceDBStorage
            from crewai.memory.unified_memory import Memory

            storage = LanceDBStorage(path=storage_path) if storage_path else LanceDBStorage()
            embedder = None
            if embedder_config is not None:
                from crewai.rag.embeddings.factory import build_embedder

                embedder = build_embedder(embedder_config)
            self._memory = Memory(storage=storage, embedder=embedder) if embedder else Memory(storage=storage)
        except Exception as e:
            self._init_error = str(e)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            yield self._build_scope_tree()
            initial = (
                self._init_error
                if self._init_error
                else "Select a scope or type a recall query."
            )
            with Vertical(id="right-panel"):
                yield Static(initial, id="info-panel")
                yield OptionList(id="entry-list")
        yield Input(
            placeholder="Type a query and press Enter to recall...",
            id="recall-input",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Set initial border titles on mounted widgets."""
        self.query_one("#info-panel", Static).border_title = "Detail"
        self.query_one("#entry-list", OptionList).border_title = "Entries"

    def _build_scope_tree(self) -> Tree[str]:
        tree: Tree[str] = Tree("/", id="scope-tree")
        if self._memory is None:
            tree.root.data = "/"
            tree.root.label = "/ (0 records)"
            return tree
        info = self._memory.info("/")
        tree.root.label = f"/ ({info.record_count} records)"
        tree.root.data = "/"
        self._add_children(tree.root, "/", depth=0, max_depth=3)
        tree.root.expand()
        return tree

    def _add_children(
        self,
        parent_node: Tree.Node[str],
        path: str,
        depth: int,
        max_depth: int,
    ) -> None:
        if depth >= max_depth or self._memory is None:
            return
        info = self._memory.info(path)
        for child in info.child_scopes:
            child_info = self._memory.info(child)
            label = f"{child} ({child_info.record_count})"
            node = parent_node.add(label, data=child)
            self._add_children(node, child, depth + 1, max_depth)

    # -- Populating the OptionList -------------------------------------------

    def _populate_entry_list(self) -> None:
        """Clear the OptionList and fill it with the current scope's entries."""
        option_list = self.query_one("#entry-list", OptionList)
        option_list.clear_options()
        for record in self._entries:
            date_str = record.created_at.strftime("%Y-%m-%d")
            preview = (
                (record.content[:80] + "…")
                if len(record.content) > 80
                else record.content
            )
            label = (
                f"{date_str}  "
                f"[bold]{record.importance:.1f}[/]  "
                f"{preview}"
            )
            option_list.add_option(label)

    def _populate_recall_list(self) -> None:
        """Clear the OptionList and fill it with the current recall matches."""
        option_list = self.query_one("#entry-list", OptionList)
        option_list.clear_options()
        if not self._recall_matches:
            return
        for m in self._recall_matches:
            preview = (
                (m.record.content[:80] + "…")
                if len(m.record.content) > 80
                else m.record.content
            )
            label = (
                f"[bold]\\[{m.score:.2f}][/]  "
                f"{preview}  "
                f"[dim]scope={m.record.scope}[/]"
            )
            option_list.add_option(label)

    # -- Detail rendering ----------------------------------------------------

    def _format_record_detail(self, record: Any, context_line: str = "") -> str:
        """Format a full MemoryRecord as Rich markup for the detail view.

        Args:
            record: A MemoryRecord instance.
            context_line: Optional header line shown above the fields
                (e.g. "Entry 3 of 47").

        Returns:
            A Rich-markup string with all meaningful record fields.
        """
        sep = f"[bold {_PRIMARY}]{'─' * 44}[/]"
        lines: list[str] = []

        if context_line:
            lines.append(context_line)
            lines.append("")

        # -- Fields block --
        lines.append(f"[dim]ID:[/]             {record.id}")
        lines.append(f"[dim]Scope:[/]          [bold]{record.scope}[/]")
        lines.append(f"[dim]Importance:[/]      [bold]{record.importance:.2f}[/]")
        lines.append(
            f"[dim]Created:[/]        "
            f"{record.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(
            f"[dim]Last accessed:[/]  "
            f"{record.last_accessed.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append(
            f"[dim]Categories:[/]     "
            f"{', '.join(record.categories) if record.categories else 'none'}"
        )
        lines.append(f"[dim]Source:[/]         {record.source or '-'}")
        lines.append(f"[dim]Private:[/]        {'Yes' if record.private else 'No'}")

        # -- Content block --
        lines.append(f"\n{sep}")
        lines.append("[bold]Content[/]\n")
        lines.append(record.content)

        # -- Metadata block --
        if record.metadata:
            lines.append(f"\n{sep}")
            lines.append("[bold]Metadata[/]\n")
            for k, v in record.metadata.items():
                lines.append(f"[dim]{k}:[/] {v}")

        return "\n".join(lines)

    # -- Event handlers ------------------------------------------------------

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        """Load entries for the selected scope and populate the OptionList."""
        path = event.node.data if event.node.data is not None else "/"
        self._selected_scope = path
        self._view_mode = "list"
        panel = self.query_one("#info-panel", Static)
        if self._memory is None:
            panel.update(self._init_error or "No memory loaded.")
            return
        _DISPLAY_LIMIT = 1000
        info = self._memory.info(path)
        self._last_scope_info = info
        self._entries = self._memory.list_records(scope=path, limit=_DISPLAY_LIMIT)
        panel.update(_format_scope_info(info))
        panel.border_title = "Detail"
        entry_list = self.query_one("#entry-list", OptionList)
        capped = info.record_count > _DISPLAY_LIMIT
        count_label = (
            f"Entries (showing {_DISPLAY_LIMIT} of {info.record_count} — display limit)"
            if capped
            else f"Entries ({len(self._entries)})"
        )
        entry_list.border_title = count_label
        self._populate_entry_list()

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        """Live-update the info panel with the detail of the highlighted entry."""
        panel = self.query_one("#info-panel", Static)
        idx = event.option_index

        if self._view_mode == "list":
            if idx < len(self._entries):
                record = self._entries[idx]
                total = len(self._entries)
                context = (
                    f"[bold {_PRIMARY}]Entry {idx + 1} of {total}[/]  "
                    f"[dim]in[/] [bold]{self._selected_scope}[/]"
                )
                panel.border_title = f"Entry {idx + 1} of {total}"
                panel.update(self._format_record_detail(record, context_line=context))

        elif self._view_mode == "recall":
            if idx < len(self._recall_matches):
                match = self._recall_matches[idx]
                total = len(self._recall_matches)
                panel.border_title = f"Match {idx + 1} of {total}"
                score_color = _PRIMARY if match.score >= 0.5 else "dim"
                header_lines: list[str] = [
                    f"[bold {_PRIMARY}]Recall Match {idx + 1} of {total}[/]\n",
                    f"[dim]Score:[/]          [{score_color}][bold]{match.score:.2f}[/][/]",
                    (
                        f"[dim]Match reasons:[/]  "
                        f"{', '.join(match.match_reasons) if match.match_reasons else '-'}"
                    ),
                    (
                        f"[dim]Evidence gaps:[/]  "
                        f"{', '.join(match.evidence_gaps) if match.evidence_gaps else 'none'}"
                    ),
                    f"\n[bold {_PRIMARY}]{'─' * 44}[/]",
                ]
                record_detail = self._format_record_detail(match.record)
                header_lines.append(record_detail)
                panel.update("\n".join(header_lines))

    def on_input_submitted(self, event: Input.Submitted) -> None:
        query = event.value.strip()
        if not query:
            return
        if self._memory is None:
            panel = self.query_one("#info-panel", Static)
            panel.update(self._init_error or "No memory loaded. Cannot recall.")
            return
        self.run_worker(self._do_recall(query), exclusive=True)

    async def _do_recall(self, query: str) -> None:
        """Execute a recall query and display results in the OptionList."""
        panel = self.query_one("#info-panel", Static)
        panel.loading = True
        try:
            scope = (
                self._selected_scope
                if self._selected_scope != "/"
                else None
            )
            loop = asyncio.get_event_loop()
            matches = await loop.run_in_executor(
                None,
                lambda: self._memory.recall(
                    query, scope=scope, limit=10, depth="deep"
                ),
            )
            self._recall_matches = matches or []
            self._view_mode = "recall"

            if not self._recall_matches:
                panel.update("[dim]No memories found.[/]")
                self.query_one("#entry-list", OptionList).clear_options()
                return

            info_lines: list[str] = []
            info_lines.append(
                "[dim italic]Searched the full dataset"
                + (f" within [bold]{scope}[/]" if scope else "")
                + " using the recall flow (semantic + recency + importance).[/]\n"
            )
            if not self._custom_embedder:
                info_lines.append(
                    "[dim italic]Note: Using default OpenAI embedder. "
                    "If memories were created with a different embedder, "
                    "pass --embedder-provider to match.[/]\n"
                )
            info_lines.append(
                f"[bold]Recall Results[/] [dim]"
                f"({len(self._recall_matches)} matches)[/]\n"
                f"[dim]Navigate the list below to view details.[/]"
            )
            panel.update("\n".join(info_lines))
            panel.border_title = "Recall Detail"
            entry_list = self.query_one("#entry-list", OptionList)
            entry_list.border_title = f"Recall Results ({len(self._recall_matches)})"
            self._populate_recall_list()
        except Exception as e:
            panel.update(f"[bold red]Error:[/] {e}")
        finally:
            panel.loading = False
