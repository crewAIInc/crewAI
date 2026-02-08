"""Textual TUI for browsing and recalling unified memory."""

from __future__ import annotations

from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Input, Static, Tree


def _format_scope_info(info: Any) -> str:
    """Format ScopeInfo as plain text."""
    return (
        f"Scope: {info.path}\n"
        f"Records: {info.record_count}\n"
        f"Categories: {', '.join(info.categories) or 'none'}\n"
        f"Oldest: {info.oldest_record or '-'}\n"
        f"Newest: {info.newest_record or '-'}\n"
        f"Child scopes: {', '.join(info.child_scopes) or 'none'}"
    )


class MemoryTUI(App[None]):
    """TUI to browse memory scopes and run recall queries."""

    PAGE_SIZE = 20

    BINDINGS = [
        ("n", "next_page", "Next page"),
        ("p", "prev_page", "Prev page"),
    ]

    CSS = """
    Horizontal { height: 1fr; }
    #scope-tree { width: 30%; }
    #info-panel { width: 70%; overflow-y: auto; }
    Input { dock: bottom; }
    """

    def __init__(self, storage_path: str | None = None) -> None:
        super().__init__()
        self._memory: Any = None
        self._init_error: str | None = None
        self._selected_scope: str = "/"
        self._entries: list[Any] = []
        self._page: int = 0
        self._last_scope_info: Any = None
        path = storage_path or "./.crewai/memory"
        try:
            from crewai.memory.storage.lancedb_storage import LanceDBStorage
            from crewai.memory.unified_memory import Memory

            storage = LanceDBStorage(path=path)
            self._memory = Memory(storage=storage)
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
            yield Static(initial, id="info-panel")
        yield Input(
            placeholder="Type a query and press Enter to recall...",
            id="recall-input",
        )
        yield Footer()

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

    def on_tree_node_selected(self, event: Tree.NodeSelected[str]) -> None:
        path = event.node.data if event.node.data is not None else "/"
        self._selected_scope = path
        panel = self.query_one("#info-panel", Static)
        if self._memory is None:
            panel.update(self._init_error or "No memory loaded.")
            return
        info = self._memory.info(path)
        self._last_scope_info = info
        self._entries = self._memory.list_records(scope=path, limit=200)
        self._page = 0
        self._render_panel()

    def _render_panel(self) -> None:
        """Refresh info panel with scope info and current page of entries."""
        panel = self.query_one("#info-panel", Static)
        if self._last_scope_info is None:
            return
        lines: list[str] = [_format_scope_info(self._last_scope_info)]
        total = len(self._entries)
        if total == 0:
            lines.append("\n--- No entries ---")
        else:
            total_pages = (total + self.PAGE_SIZE - 1) // self.PAGE_SIZE
            page_num = self._page + 1
            lines.append(f"\n--- Entries (page {page_num} of {total_pages}) ---")
            start = self._page * self.PAGE_SIZE
            end = min(start + self.PAGE_SIZE, total)
            for record in self._entries[start:end]:
                date_str = record.created_at.strftime("%Y-%m-%d")
                preview = (record.content[:100] + "…") if len(record.content) > 100 else record.content
                lines.append(f"{date_str} | importance={record.importance:.1f} | {preview}")
            if total_pages > 1:
                lines.append(f"\n[n] next  [p] prev")
        panel.update("\n".join(lines))

    def action_next_page(self) -> None:
        """Go to next page of entries."""
        total_pages = (len(self._entries) + self.PAGE_SIZE - 1) // self.PAGE_SIZE
        if total_pages <= 0:
            return
        self._page = min(self._page + 1, total_pages - 1)
        self._render_panel()

    def action_prev_page(self) -> None:
        """Go to previous page of entries."""
        if self._page <= 0:
            return
        self._page -= 1
        self._render_panel()

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
        panel = self.query_one("#info-panel", Static)
        panel.update(f"Recalling: {query} ...")
        try:
            scope = (
                self._selected_scope
                if self._selected_scope != "/"
                else None
            )
            matches = self._memory.recall(
                query,
                scope=scope,
                limit=10,
                depth="shallow",
            )
            if not matches:
                panel.update("No memories found.")
                return
            lines: list[str] = []
            for m in matches:
                content = m.record.content
                lines.append(f"[{m.score:.2f}] {content[:120]}")
                lines.append(
                    f"       scope={m.record.scope}  "
                    f"importance={m.record.importance:.1f}"
                )
                lines.append("")
            panel.update("\n".join(lines))
        except Exception as e:
            panel.update(f"Error: {e}")
