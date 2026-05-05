"""Minimal Textual TUI for triaging skill proposals.

Two panes: the proposals list on the left, the highlighted proposal's
``SKILL.md`` body on the right. Keystrokes accept/reject in place. No
search, no scopes, no async workers — the underlying actions are the
same `accept_proposal` / `reject_proposal` calls the CLI uses.
"""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Footer, Header, OptionList, Static

from crewai.skills.self_improve import (
    ProposalStore,
    accept_proposal,
    reject_proposal,
)
from crewai.skills.self_improve.models import SkillProposal


_PRIMARY = "#eb6658"
_SECONDARY = "#1F7982"
_TERTIARY = "#ffffff"


def _format_proposal_detail(p: SkillProposal) -> str:
    kind = (
        f"[bold]{p.proposal_kind}[/] → {p.target_skill}"
        if p.proposal_kind == "patch_existing"
        else "[bold]new[/]"
    )
    runs = ", ".join(p.derived_from_runs) or "-"
    return (
        f"[bold {_PRIMARY}]{p.name}[/]\n"
        f"[dim]role:[/]       {p.agent_role}\n"
        f"[dim]kind:[/]       {kind}\n"
        f"[dim]confidence:[/] [bold]{p.confidence:.2f}[/]\n"
        f"[dim]from runs:[/]  {runs}\n\n"
        f"[bold]Rationale[/]\n{p.rationale}\n\n"
        f"[bold {_PRIMARY}]{'─' * 44}[/]\n"
        f"{p.body}"
    )


class SkillProposalsTUI(App[None]):
    """Triage UI: navigate the queue, accept or reject in place."""

    TITLE = "CrewAI Skill Proposals"
    SUB_TITLE = "↑↓ list · tab focus pane · PgUp/PgDn or mouse to scroll body · a/r/q"

    BINDINGS = [
        Binding("a", "accept", "Accept"),
        Binding("r", "reject", "Reject"),
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next", "Switch pane", show=False),
    ]

    CSS = f"""
    Header {{ background: {_PRIMARY}; color: {_TERTIARY}; }}
    Footer {{ background: {_SECONDARY}; color: {_TERTIARY}; }}
    Footer > .footer-key--key {{ background: {_PRIMARY}; color: {_TERTIARY}; }}
    Horizontal {{ height: 1fr; }}
    #list {{
        width: 40%;
        border-right: solid {_SECONDARY};
        scrollbar-color: {_PRIMARY};
    }}
    #list > .option-list--option-highlighted {{
        background: {_SECONDARY}; color: {_TERTIARY};
    }}
    #detail-scroll {{
        width: 60%;
        padding: 1 2;
        scrollbar-color: {_PRIMARY};
    }}
    #detail-scroll:focus {{
        background: {_SECONDARY} 5%;
    }}
    """

    def __init__(self, store: ProposalStore | None = None) -> None:
        super().__init__()
        self._store = store or ProposalStore()
        self._proposals: list[SkillProposal] = []

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal():
            yield OptionList(id="list")
            with VerticalScroll(id="detail-scroll"):
                yield Static("Select a proposal to view its body.", id="detail")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#list", OptionList).border_title = "Pending"
        self.query_one("#detail-scroll", VerticalScroll).border_title = "Detail"
        self._reload()

    def _reload(self) -> None:
        self._proposals = self._store.list_all()
        option_list = self.query_one("#list", OptionList)
        option_list.clear_options()
        for p in self._proposals:
            kind_tag = "P" if p.proposal_kind == "patch_existing" else "N"
            label = f"[{kind_tag}] {p.confidence:.2f}  {p.name}"
            option_list.add_option(label)
        option_list.border_title = f"Pending ({len(self._proposals)})"
        if not self._proposals:
            self.query_one("#detail", Static).update("[dim](queue is empty)[/]")

    def _selected(self) -> SkillProposal | None:
        idx = self.query_one("#list", OptionList).highlighted
        if idx is None or idx >= len(self._proposals):
            return None
        return self._proposals[idx]

    def on_option_list_option_highlighted(
        self, event: OptionList.OptionHighlighted
    ) -> None:
        idx = event.option_index
        if idx < len(self._proposals):
            self.query_one("#detail", Static).update(
                _format_proposal_detail(self._proposals[idx])
            )

    def action_accept(self) -> None:
        prop = self._selected()
        if prop is None:
            return
        try:
            accept_proposal(prop)
            self.notify(f"Accepted: {prop.name}", severity="information")
        except FileExistsError as e:
            self.notify(str(e), severity="warning", timeout=8)
            return
        self._reload()

    def action_reject(self) -> None:
        prop = self._selected()
        if prop is None:
            return
        reject_proposal(prop)
        self.notify(f"Rejected: {prop.name}", severity="information")
        self._reload()
