"""Flow panels must be suppressed while a TUI owns the screen."""

from rich.text import Text

from crewai.events.listeners.tracing.utils import set_tui_mode
from crewai.events.utils.console_formatter import ConsoleFormatter


def _make_formatter(monkeypatch):
    fmt = ConsoleFormatter(verbose=True)
    calls: list[object] = []
    monkeypatch.setattr(fmt, "print", lambda *a, **k: calls.append(a))
    return fmt, calls


def test_flow_panel_suppressed_in_tui_mode(monkeypatch):
    fmt, calls = _make_formatter(monkeypatch)
    set_tui_mode(True)
    try:
        fmt.print_panel(Text("x"), "🌊 Flow Started", "blue", is_flow=True)
    finally:
        set_tui_mode(False)

    assert calls == []


def test_flow_panel_prints_when_not_tui_mode(monkeypatch):
    fmt, calls = _make_formatter(monkeypatch)
    set_tui_mode(False)

    fmt.print_panel(Text("x"), "🌊 Flow Started", "blue", is_flow=True)

    # Panel + trailing blank line.
    assert len(calls) == 2


def test_non_flow_panel_unaffected_by_tui_mode(monkeypatch):
    # tui_mode only gates flow panels; regular panels still follow verbose.
    fmt, calls = _make_formatter(monkeypatch)
    set_tui_mode(True)
    try:
        fmt.print_panel(Text("x"), "Task", "blue", is_flow=False)
    finally:
        set_tui_mode(False)

    assert len(calls) == 2
