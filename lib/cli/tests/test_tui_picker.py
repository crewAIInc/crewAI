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


def test_searchable_draw_shows_the_query_and_an_empty_match(monkeypatch) -> None:
    output = StringIO()
    monkeypatch.setattr(tui_picker.sys, "stdout", output)

    tui_picker._draw_multi(
        ["Send an email (send-email)"],
        cursor=0,
        selected={0},
        row_indices=[],
        query="missing",
        searchable=True,
    )

    rendered = output.getvalue()
    assert "Search:" in rendered
    assert "missing" in rendered
    assert "type to filter" in rendered
    assert "No matching actions" in rendered
    assert "Send an email" not in rendered


def test_numbered_search_keeps_original_indexes(monkeypatch) -> None:
    prompts = iter(["draft", "1"])
    monkeypatch.setattr(
        tui_picker.click, "prompt", lambda *_args, **_kwargs: next(prompts)
    )

    selected, action = tui_picker._numbered_select_multi(
        [
            "Send an email (send-email)",
            "Create email draft (create_draft)",
            "Search emails (search-emails)",
        ],
        preselected={0},
        searchable=True,
    )

    assert selected == [0, 1]
    assert action is None


def test_searchable_picker_keeps_preselected_rows_while_filtering(monkeypatch) -> None:
    keys = iter(["d", "r", "a", "f", "t", "space", "enter"])
    monkeypatch.setattr(tui_picker, "_read_key", lambda: next(keys))
    monkeypatch.setattr(tui_picker, "_clear_lines", lambda *_args, **_kwargs: None)
    labels = [
        "Send an email (send-email)",
        "Create email draft (create_draft)",
        "Search emails (search-emails)",
    ]

    assert tui_picker._arrow_select_multi(
        labels, preselected={0}, searchable=True
    ) == ([0, 1], None)


def test_searchable_picker_clears_the_query_on_escape(monkeypatch) -> None:
    keys = iter(["f", "esc", "up", "space", "enter"])
    monkeypatch.setattr(tui_picker, "_read_key", lambda: next(keys))
    monkeypatch.setattr(tui_picker, "_clear_lines", lambda *_args, **_kwargs: None)

    assert tui_picker._arrow_select_multi(
        ["Send an email (send-email)", "Create email draft (create_draft)"],
        searchable=True,
    ) == ([0], None)


def test_matching_label_indices_keeps_original_positions() -> None:
    labels = [
        "Send an email (send-email)",
        "Create email draft (create_draft)",
        "Search emails (search-emails)",
    ]

    assert tui_picker.matching_label_indices(labels, "") == [0, 1, 2]
    assert tui_picker.matching_label_indices(labels, "EMAIL") == [0, 1, 2]
    assert tui_picker.matching_label_indices(labels, "draft") == [1]
    assert tui_picker.matching_label_indices(labels, "search-emails") == [2]
    assert tui_picker.matching_label_indices(labels, "missing") == []
