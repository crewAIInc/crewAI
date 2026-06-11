"""Arrow-key interactive pickers for CLI prompts."""

from __future__ import annotations

from contextlib import suppress
import sys

import click


# CrewAI brand: primary=#FF5A50 (coral), teal=#1F7982
_CORAL = "\033[38;2;255;90;80m"  # #FF5A50
_TEAL = "\033[38;2;31;121;130m"  # #1F7982
_GREEN = "\033[38;2;74;186;106m"  # #4aba6a
_BOLD = "\033[1m"
_DIM = "\033[2m"
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
) -> None:
    action_indices = action_indices or set()
    separator_indices = separator_indices or set()
    hint_text = "↑↓ navigate, space toggle, enter confirm"
    if action_indices:
        hint_text = "↑↓ navigate, space toggle, enter confirm, space/enter opens > rows"
    hint = f"  {_DIM}{hint_text}{_RESET}"
    total = len(labels) + 1
    if clear:
        sys.stdout.write(f"\033[{total}A")
    sys.stdout.write(f"\033[2K{hint}\n")
    for i, label in enumerate(labels):
        if i in separator_indices:
            sys.stdout.write(f"\033[2K      {_TEAL}{label}{_RESET}\n")
            continue
        if i in action_indices:
            check = f"{_TEAL}>{_RESET}  "
        elif i in selected:
            check = f"{_CORAL}[x]{_RESET}"
        else:
            check = "[ ]"
        arrow = f"{_CORAL}→{_RESET} " if i == cursor else "  "
        bold = f"{_BOLD}{label}{_RESET}" if i == cursor else label
        sys.stdout.write(f"\033[2K    {arrow}{check} {bold}\n")
    sys.stdout.flush()


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


def _arrow_select_multi(
    labels: list[str],
    *,
    action_indices: set[int] | None = None,
    separator_indices: set[int] | None = None,
) -> tuple[list[int], int | None]:
    total = len(labels)
    selected: set[int] = set()
    action_indices = action_indices or set()
    separator_indices = separator_indices or set()
    cursor = _first_selectable_index(total, separator_indices)
    sys.stdout.write(_HIDE_CURSOR)
    sys.stdout.flush()
    try:
        _draw_multi(
            labels,
            cursor,
            selected,
            action_indices=action_indices,
            separator_indices=separator_indices,
        )
        while True:
            key = _read_key()
            if key == "up":
                cursor = _next_selectable_index(cursor, -1, total, separator_indices)
                _draw_multi(
                    labels,
                    cursor,
                    selected,
                    action_indices=action_indices,
                    separator_indices=separator_indices,
                    clear=True,
                )
            elif key == "down":
                cursor = _next_selectable_index(cursor, 1, total, separator_indices)
                _draw_multi(
                    labels,
                    cursor,
                    selected,
                    action_indices=action_indices,
                    separator_indices=separator_indices,
                    clear=True,
                )
            elif key == "space":
                if cursor in action_indices:
                    _clear_lines(total + 1)
                    return sorted(selected), cursor
                selected ^= {cursor}
                _draw_multi(
                    labels,
                    cursor,
                    selected,
                    action_indices=action_indices,
                    separator_indices=separator_indices,
                    clear=True,
                )
            elif key == "enter":
                _clear_lines(total + 1)
                if cursor in action_indices:
                    return sorted(selected), cursor
                return sorted(selected), None
            elif key in ("esc", "q"):
                _clear_lines(total + 1)
                return [], None
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
            pass
        click.secho(f"  Invalid choice. Enter 1-{len(labels)}.", fg="red")


def _numbered_select_multi(
    labels: list[str],
    *,
    action_indices: set[int] | None = None,
    separator_indices: set[int] | None = None,
) -> tuple[list[int], int | None]:
    action_indices = action_indices or set()
    separator_indices = separator_indices or set()
    numbered_indices: list[int] = []
    for idx, label in enumerate(labels):
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
        return [], None
    indices = []
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


def pick_many(
    title: str,
    labels: list[str],
    *,
    action_indices: set[int] | None = None,
    separator_indices: set[int] | None = None,
) -> list[int] | tuple[list[int], int | None]:
    """Arrow-key multi-select with checkboxes.

    Returns:
        Sorted list of selected indices, or ``(indices, action_index)`` when
        ``action_indices`` is provided.
    """
    click.echo()
    click.secho(f"  {title}", fg="cyan")

    if _is_interactive():
        try:
            selected, action = _arrow_select_multi(
                labels,
                action_indices=action_indices,
                separator_indices=separator_indices,
            )
        except Exception:
            selected, action = _numbered_select_multi(
                labels,
                action_indices=action_indices,
                separator_indices=separator_indices,
            )
    else:
        selected, action = _numbered_select_multi(
            labels,
            action_indices=action_indices,
            separator_indices=separator_indices,
        )
    if action_indices is None:
        return selected
    return selected, action
