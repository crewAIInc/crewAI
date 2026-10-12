import os
from io import StringIO

from crewai_cli import tui_picker


def test_visible_row_range_keeps_short_lists_fully_visible(monkeypatch) -> None:
    monkeypatch.setattr(
        tui_picker.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((80, 24)),
    )

    assert tui_picker._visible_row_range(total=8, cursor=3) == (0, 8)


def test_visible_row_range_scrolls_long_lists_around_cursor(monkeypatch) -> None:
    monkeypatch.setattr(
        tui_picker.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((80, 12)),
    )

    assert tui_picker._visible_row_range(total=125, cursor=0) == (0, 7)
    assert tui_picker._visible_row_range(total=125, cursor=62) == (59, 66)
    assert tui_picker._visible_row_range(total=125, cursor=124) == (118, 125)


def test_draw_multi_renders_only_the_visible_window(monkeypatch) -> None:
    monkeypatch.setattr(
        tui_picker.shutil,
        "get_terminal_size",
        lambda fallback: os.terminal_size((80, 12)),
    )
    output = StringIO()
    monkeypatch.setattr(tui_picker.sys, "stdout", output)

    line_count = tui_picker._draw_multi(
        [f"Tool {index}" for index in range(125)],
        cursor=62,
        selected=set(),
    )

    rendered = output.getvalue()
    assert line_count == 8
    assert "showing 60-66 of 125" in rendered
    assert "Tool 59" in rendered
    assert "Tool 65" in rendered
    assert "Tool 0" not in rendered
    assert "Tool 124" not in rendered
