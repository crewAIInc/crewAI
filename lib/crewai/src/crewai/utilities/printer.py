"""Utility for colored console output."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, NamedTuple


if TYPE_CHECKING:
    from _typeshed import SupportsWrite

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
    """Represents text with an optional color for console output.

    Attributes:
        text: The text content to be printed.
        color: Optional color for the text, specified as a PrinterColor.
    """

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
        """Prints content to the console with optional color formatting.

        Args:
            content: Either a string or a list of ColoredText objects for multicolor output.
            color: Optional color for the text when content is a string. Ignored when content is a list.
            sep: Separator to use between the text and color.
            end: String appended after the last value.
            file: A file-like object (stream); defaults to the current sys.stdout.
            flush: Whether to forcibly flush the stream.
        """
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
