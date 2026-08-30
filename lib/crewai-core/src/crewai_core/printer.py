"""Colored console-output utilities and the shared output-suppression flag."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING, Final, Literal, NamedTuple


if TYPE_CHECKING:
    from _typeshed import SupportsWrite


_suppress_console_output: ContextVar[bool] = ContextVar(
    "_suppress_console_output", default=False
)


def set_suppress_console_output(suppress: bool) -> object:
    """Toggle suppression of console output for the current context.

    Returns a token that can be passed to ``ContextVar.reset`` to restore the
    previous value.
    """
    return _suppress_console_output.set(suppress)


def should_suppress_console_output() -> bool:
    """Return True if console output should currently be suppressed."""
    return _suppress_console_output.get()


PrinterColor = Literal[
    "purple",
    "bold_purple",
    "green",
    "bold_green",
    "cyan",
    "bold_cyan",
    "magenta",
    "bold_magenta",
    "yellow",
    "bold_yellow",
    "red",
    "blue",
    "bold_blue",
]

_COLOR_CODES: Final[dict[PrinterColor, str]] = {
    "purple": "\033[95m",
    "bold_purple": "\033[1m\033[95m",
    "red": "\033[91m",
    "bold_green": "\033[1m\033[92m",
    "green": "\033[32m",
    "blue": "\033[94m",
    "bold_blue": "\033[1m\033[94m",
    "yellow": "\033[93m",
    "bold_yellow": "\033[1m\033[93m",
    "cyan": "\033[96m",
    "bold_cyan": "\033[1m\033[96m",
    "magenta": "\033[35m",
    "bold_magenta": "\033[1m\033[35m",
}

RESET: Final[str] = "\033[0m"


class ColoredText(NamedTuple):
    """Text plus an optional color, used for multicolor lines."""

    text: str
    color: PrinterColor | None


class Printer:
    """Handles colored console output formatting."""

    @staticmethod
    def print(
        content: str | list[ColoredText],
        color: PrinterColor | None = None,
        sep: str | None = " ",
        end: str | None = "\n",
        file: SupportsWrite[str] | None = None,
        flush: Literal[False] = False,
    ) -> None:
        """Print ``content`` with optional color, honoring suppression context."""
        if should_suppress_console_output():
            return
        if isinstance(content, str):
            content = [ColoredText(content, color)]
        print(
            "".join(
                f"{_COLOR_CODES[c.color] if c.color else ''}{c.text}{RESET}"
                for c in content
            ),
            sep=sep,
            end=end,
            file=file,
            flush=flush,
        )


PRINTER: Printer = Printer()
