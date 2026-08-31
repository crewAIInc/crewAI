"""Parse ``/btw`` interjections from a conversational user line.

A leading ``/btw`` is a side-channel command: it steers later turns and
does not become the user utterance. Appending `` /btw …`` to a normal
message applies the command and still runs the remaining text as a turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


_LEADING_BTW = re.compile(r"^/btw(?:\s+|$)", re.IGNORECASE)
_INLINE_BTW = re.compile(r"\s+/btw(?:\s+|$)", re.IGNORECASE)
_HELP_LINES = frozenset({"/help", "/?"})

HELP_TEXT = (
    "Experimental /btw commands (side-channel; they do not become a user turn):\n"
    "  /btw <note>                    Persist a steering note for later turns\n"
    "  /btw route <label>             Force the next turn onto that route\n"
    "  /btw route <label> persist     Keep forcing that route until /btw clear\n"
    "  /btw stay <label>              Same as route <label> persist\n"
    "  /btw clear                     Drop notes and forced routes\n"
    "  /btw show                      Show current steering\n"
    "  /help                          This list\n"
    "\n"
    "You can also append a command to a message:\n"
    "  What's the weather /btw keep it to one sentence"
)


class BtwKind(str, Enum):
    """Kind of ``/btw`` action parsed from a user line."""

    NOTE = "note"
    ROUTE = "route"
    CLEAR = "clear"
    SHOW = "show"
    HELP = "help"


@dataclass(frozen=True)
class BtwAction:
    """A parsed ``/btw`` (or ``/help``) action."""

    kind: BtwKind
    argument: str = ""
    persist_route: bool = False


@dataclass(frozen=True)
class ParsedBtwLine:
    """Split of a user line into an optional command and remaining utterance.

    ``user_message`` is ``None`` when the line is only a command and should
    not run a conversational turn.
    """

    action: BtwAction | None
    user_message: str | None


def parse_btw_line(message: str) -> ParsedBtwLine:
    """Parse a conversational line into a ``/btw`` action and leftover text."""
    stripped = message.strip()
    if not stripped:
        return ParsedBtwLine(action=None, user_message=stripped)

    if stripped.lower() in _HELP_LINES:
        return ParsedBtwLine(action=BtwAction(kind=BtwKind.HELP), user_message=None)

    leading = _LEADING_BTW.match(stripped)
    if leading is not None:
        return ParsedBtwLine(
            action=_parse_btw_body(stripped[leading.end() :].strip()),
            user_message=None,
        )

    inline = _INLINE_BTW.search(stripped)
    if inline is not None:
        user = stripped[: inline.start()].strip()
        return ParsedBtwLine(
            action=_parse_btw_body(stripped[inline.end() :].strip()),
            user_message=user or None,
        )

    return ParsedBtwLine(action=None, user_message=stripped)


def _parse_btw_body(body: str) -> BtwAction:
    if not body:
        return BtwAction(kind=BtwKind.SHOW)

    parts = body.split()
    head = parts[0].lower()
    if head == "clear":
        return BtwAction(kind=BtwKind.CLEAR)
    if head in {"show", "status"}:
        return BtwAction(kind=BtwKind.SHOW)
    if head == "help":
        return BtwAction(kind=BtwKind.HELP)
    if head == "route" and len(parts) >= 2:
        persist = len(parts) >= 3 and parts[2].lower() in {"persist", "stay"}
        return BtwAction(kind=BtwKind.ROUTE, argument=parts[1], persist_route=persist)
    if head == "stay" and len(parts) >= 2:
        return BtwAction(kind=BtwKind.ROUTE, argument=parts[1], persist_route=True)
    return BtwAction(kind=BtwKind.NOTE, argument=body)
