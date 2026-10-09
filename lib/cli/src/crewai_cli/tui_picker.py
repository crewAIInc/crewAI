"""Arrow-key interactive pickers for CLI prompts."""

from __future__ import annotations

from contextlib import suppress
import shutil
import sys
from typing import overload

import click


# CrewAI brand: primary=#FF5A50 (coral), teal=#1F7982
_CORAL = "\033[38;2;255;90;80m"  # #FF5A50
_TEAL = "\033[38;2;31;121;130m"  # #1F7982
_BOLD = "\033[1m"
_DIM = "\033[2m"
_GREEN = "\033[1;32m"
_RESET = "\033[0m"
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"


def _is_interactive() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _read_key() -> str:
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            return {"H": "up", "P": "down"}.get(ch2, "")
        if ch == "\r":
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\x03":
            raise KeyboardInterrupt
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _clear_lines(n: int) -> None:
    sys.stdout.write(f"\033[{n}A")
    for _ in range(n):
        sys.stdout.write("\033[2K\n")
    sys.stdout.write(f"\033[{n}A")
    sys.stdout.flush()


def _draw_single(labels: list[str], cursor: int, *, clear: bool = False) -> None:
    total = len(labels)
    if clear:
        sys.stdout.write(f"\033[{total}A")
    for i, label in enumerate(labels):
        if i == cursor:
            sys.stdout.write(f"\033[2K  {_CORAL}→{_RESET} {_BOLD}{label}{_RESET}\n")
        else:
            sys.stdout.write(f"\033[2K    {label}\n")
    sys.stdout.flush()


def _draw_multi(
    labels: list[str],
    cursor: int,
    selected: set[int],
    *,
    action_indices: set[int] | None = None,
    separator_indices: set[int] | None = None,
    clear: bool = False,
    previous_line_count: int | None = None,
    row_indices: list[int] | None = None,
    query: str = "",
    searchable: bool = False,
) -> int:
    action_indices = action_indices or set()
    separator_indices = separator_indices or set()
    rows = row_indices if row_indices is not None else list(range(len(labels)))
    try:
        cursor_pos = rows.index(cursor)
    except ValueError:
        cursor_pos = 0
    start, end = _visible_row_range(len(rows), cursor_pos)
    hint_text = "↑↓ navigate, space toggle, enter confirm"
    if action_indices:
        hint_text = "↑↓ navigate, space toggle, enter confirm, ▸ rows expand/collapse"
    if searchable:
        hint_text += " · type to filter"
    if end - start < len(rows):
        hint_text += f" · showing {start + 1}-{end} of {len(rows)}"
    hint = f"  {_DIM}{hint_text}{_RESET}"
    if clear:
        # Erase the previous block first. A shorter search result would
        # otherwise leave the old rows visible underneath the matches.
        _clear_lines(previous_line_count or 1)
    extra_lines = 0
    sys.stdout.write(f"\033[2K{hint}\n")
    if searchable:
        match_count = ""
        if query:
            count = len(rows)
            noun = "match" if count == 1 else "matches"
            match_count = f"  ·  {count} {noun}"
        sys.stdout.write(f"\033[2K  {_GREEN}Search: {query}{match_count}{_RESET}\n")
        extra_lines += 1
    if searchable and query and not rows:
        sys.stdout.write(f"\033[2K    {_DIM}No matching actions{_RESET}\n")
        extra_lines += 1
    for pos in range(start, end):
        i = rows[pos]
        label = labels[i]
        if i in separator_indices:
            sys.stdout.write(f"\033[2K      {_TEAL}{label}{_RESET}\n")
            continue
        if i in action_indices:
            check = "  "
        elif i in selected:
            check = f"{_CORAL}[x]{_RESET}"
        else:
            check = "[ ]"
        arrow = f"{_CORAL}→{_RESET} " if i == cursor else "  "
        bold = f"{_BOLD}{label}{_RESET}" if i == cursor else label
        sys.stdout.write(f"\033[2K    {arrow}{check} {bold}\n")
    sys.stdout.flush()
    return end - start + 1 + extra_lines


def _visible_row_range(total: int, cursor: int) -> tuple[int, int]:
    """Return the portion of a multi-select list that fits in the terminal."""
    max_rows = max(5, shutil.get_terminal_size(fallback=(80, 24)).lines - 5)
    if total <= max_rows:
        return 0, total

    start = min(max(cursor - max_rows // 2, 0), total - max_rows)
    return start, start + max_rows


def _arrow_select_one(labels: list[str]) -> int:
    cursor = 0
    total = len(labels)
    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()
    try:
        _draw_single(labels, cursor)
        while True:
            key = _read_key()
            if key == "up" and cursor > 0:
                cursor -= 1
                _draw_single(labels, cursor, clear=True)
            elif key == "down" and cursor < total - 1:
                cursor += 1
                _draw_single(labels, cursor, clear=True)
            elif key == "enter":
                _clear_lines(total)
                return cursor
            elif key in ("esc", "q"):
                _clear_lines(total)
                return -1
    finally:
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()


def _is_search_character(key: str) -> bool:
    return len(key) == 1 and key.isprintable() and key != " "


def _visible_label_indices(
    labels: list[str],
    query: str,
    separator_indices: set[int],
    *,
    searchable: bool,
) -> list[int]:
    if not searchable or not query:
        return list(range(len(labels)))
    return [
        index
        for index in matching_label_indices(labels, query)
        if index not in separator_indices
    ]


def _cursor_on_visible(
    cursor: int,
    labels: list[str],
    query: str,
    separator_indices: set[int],
    searchable: bool,
) -> int:
    visible = _visible_label_indices(
        labels, query, separator_indices, searchable=searchable
    )
    if cursor in visible or not visible:
        return cursor
    return visible[0]


def _move_visible_cursor(
    cursor: int,
    direction: int,
    labels: list[str],
    query: str,
    separator_indices: set[int],
    searchable: bool,
) -> int:
    visible = [
        index
        for index in _visible_label_indices(
            labels, query, separator_indices, searchable=searchable
        )
        if index not in separator_indices
    ]
    if cursor not in visible:
        return visible[0] if visible else cursor
    position = visible.index(cursor) + direction
    if 0 <= position < len(visible):
        return visible[position]
    return cursor


def _arrow_select_multi(
    labels: list[str],
    *,
    action_indices: set[int] | None = None,
    separator_indices: set[int] | None = None,
    preselected: set[int] | None = None,
    initial_cursor: int | None = None,
    searchable: bool = False,
) -> tuple[list[int], int | None]:
    total = len(labels)
    selected: set[int] = set(preselected or ())
    action_indices = action_indices or set()
    separator_indices = separator_indices or set()
    query = ""
    if initial_cursor is not None and 0 <= initial_cursor < total:
        cursor = initial_cursor
    else:
        cursor = _first_selectable_index(total, separator_indices)
    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()

    def redraw(clear: bool) -> int:
        return _draw_multi(
            labels,
            cursor,
            selected,
            action_indices=action_indices,
            separator_indices=separator_indices,
            clear=clear,
            previous_line_count=rendered_lines if clear else None,
            row_indices=_visible_label_indices(
                labels, query, separator_indices, searchable=searchable
            ),
            query=query,
            searchable=searchable,
        )

    try:
        rendered_lines = redraw(False)
        while True:
            key = _read_key()
            if key == "up":
                cursor = _move_visible_cursor(
                    cursor, -1, labels, query, separator_indices, searchable
                )
            elif key == "down":
                cursor = _move_visible_cursor(
                    cursor, 1, labels, query, separator_indices, searchable
                )
            elif key == "space":
                if cursor in action_indices:
                    _clear_lines(rendered_lines)
                    return sorted(selected), cursor
                if cursor in _visible_label_indices(
                    labels, query, separator_indices, searchable=searchable
                ):
                    selected ^= {cursor}
            elif key == "enter":
                _clear_lines(rendered_lines)
                if cursor in action_indices:
                    return sorted(selected), cursor
                return sorted(selected), None
            elif searchable and key in ("\x7f", "\b") and query:
                query = query[:-1]
                cursor = _cursor_on_visible(
                    cursor, labels, query, separator_indices, searchable
                )
            elif searchable and key == "esc" and query:
                query = ""
                cursor = _cursor_on_visible(
                    cursor, labels, query, separator_indices, searchable
                )
            elif searchable and _is_search_character(key):
                query += key
                cursor = _cursor_on_visible(
                    cursor, labels, query, separator_indices, searchable
                )
            elif key in ("esc", "q"):
                _clear_lines(rendered_lines)
                return sorted(selected), None
            else:
                continue
            rendered_lines = redraw(True)
    finally:
        sys.stdout.write(_SHOW_CURSOR)
        sys.stdout.flush()


def _numbered_select(labels: list[str]) -> int:
    for idx, label in enumerate(labels, 1):
        click.echo(f"    {idx}. {label}")
    click.echo()
    while True:
        choice = click.prompt("  Select", type=str, default="1")
        if choice.lower() == "q":
            return -1
        try:
            num = int(choice)
            if 1 <= num <= len(labels):
                return num - 1
        except ValueError:
            # Non-numeric input falls through to the shared error message.
            pass
        click.secho(f"  Invalid choice. Enter 1-{len(labels)}.", fg="red")


def _numbered_select_multi(
    labels: list[str],
    *,
    action_indices: set[int] | None = None,
    separator_indices: set[int] | None = None,
    preselected: set[int] | None = None,
    searchable: bool = False,
) -> tuple[list[int], int | None]:
    action_indices = action_indices or set()
    separator_indices = separator_indices or set()
    query = ""
    if searchable:
        query = str(click.prompt("  Filter", default="", show_default=False)).strip()
    visible = _visible_label_indices(
        labels, query, separator_indices, searchable=searchable
    )
    numbered_indices: list[int] = []
    if searchable and query and not visible:
        click.echo("    No matching actions")
    for idx in visible:
        label = labels[idx]
        if idx in separator_indices:
            click.secho(f"    {label}", fg="cyan")
            continue
        numbered_indices.append(idx)
        click.echo(f"    {len(numbered_indices)}. {label}")
    click.echo()
    raw = click.prompt(
        "  Select (comma-separated numbers, or empty to skip)",
        default="",
        show_default=False,
    )
    if not raw.strip():
        return sorted(preselected or ()), None
    indices: list[int] = list(preselected or ())
    for part in raw.split(","):
        with suppress(ValueError):
            num = int(part.strip())
            if 1 <= num <= len(numbered_indices):
                idx = numbered_indices[num - 1]
                if idx in action_indices:
                    return sorted(set(indices)), idx
                indices.append(idx)
    return sorted(set(indices)), None


def _first_selectable_index(total: int, separator_indices: set[int]) -> int:
    for idx in range(total):
        if idx not in separator_indices:
            return idx
    return 0


def _next_selectable_index(
    cursor: int,
    direction: int,
    total: int,
    separator_indices: set[int],
) -> int:
    next_cursor = cursor + direction
    while 0 <= next_cursor < total:
        if next_cursor not in separator_indices:
            return next_cursor
        next_cursor += direction
    return cursor


def matching_label_indices(labels: list[str], query: str) -> list[int]:
    """Original indexes whose label contains ``query``, ignoring case.

    An empty query matches every label. The returned indexes stay in label
    order so a filtered view can keep the selection tied to the full list.
    """
    needle = query.casefold()
    if not needle:
        return list(range(len(labels)))
    return [index for index, label in enumerate(labels) if needle in label.casefold()]


# ── Public API ──────────────────────────────────────────────────


def pick(title: str, options: list[tuple[str, str]]) -> str | None:
    """Arrow-key single-select picker.

    Args:
        title: Header text.
        options: List of ``(value, description)`` tuples.

    Returns:
        The *value* of the selected option, or ``None`` if cancelled.
    """
    labels = [f"{value:<12s} {desc}" for value, desc in options]

    click.echo()
    click.secho(f"  {title}", fg="cyan", bold=True)
    click.echo()

    if _is_interactive():
        try:
            idx = _arrow_select_one(labels)
        except Exception:
            idx = _numbered_select(labels)
    else:
        idx = _numbered_select(labels)

    if idx < 0:
        return None

    value, _desc = options[idx]
    click.secho(f"  ✔ {value}", fg="green")
    return value


def pick_one(title: str, labels: list[str]) -> int:
    """Arrow-key single-select from plain labels.

    Returns:
        Selected index, or ``-1`` if cancelled.
    """
    click.echo()
    click.secho(f"  {title}", fg="cyan")

    if _is_interactive():
        try:
            return _arrow_select_one(labels)
        except Exception:
            return _numbered_select(labels)
    return _numbered_select(labels)


@overload
def pick_many(
    title: str,
    labels: list[str],
    *,
    separator_indices: set[int] | None = None,
    preselected: set[int] | None = None,
    initial_cursor: int | None = None,
    searchable: bool = False,
) -> list[int]: ...


@overload
def pick_many(
    title: str,
    labels: list[str],
    *,
    action_indices: set[int],
    separator_indices: set[int] | None = None,
    preselected: set[int] | None = None,
    initial_cursor: int | None = None,
    searchable: bool = False,
) -> tuple[list[int], int | None]: ...


def pick_many(
    title: str,
    labels: list[str],
    *,
    action_indices: set[int] | None = None,
    separator_indices: set[int] | None = None,
    preselected: set[int] | None = None,
    initial_cursor: int | None = None,
    searchable: bool = False,
) -> list[int] | tuple[list[int], int | None]:
    """Arrow-key multi-select with checkboxes.

    ``searchable`` lets the user type a query. Matching rows stay tied to
    their original indexes, and checked rows stay checked when filtered out.

    Returns:
        Sorted list of selected indices, or ``(indices, action_index)`` when
        ``action_indices`` is provided.
    """
    if title:
        click.echo()
        click.secho(f"  {title}", fg="cyan")

    if _is_interactive():
        try:
            selected, action = _arrow_select_multi(
                labels,
                action_indices=action_indices,
                separator_indices=separator_indices,
                preselected=preselected,
                initial_cursor=initial_cursor,
                searchable=searchable,
            )
        except Exception:
            selected, action = _numbered_select_multi(
                labels,
                action_indices=action_indices,
                separator_indices=separator_indices,
                preselected=preselected,
                searchable=searchable,
            )
    else:
        selected, action = _numbered_select_multi(
            labels,
            action_indices=action_indices,
            separator_indices=separator_indices,
            preselected=preselected,
            searchable=searchable,
        )
    if action_indices is None:
        return selected
    return selected, action
