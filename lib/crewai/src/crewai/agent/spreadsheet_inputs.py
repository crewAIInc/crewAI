"""Spreadsheets an agent is given as URLs in its input.

With the ``spreadsheets`` extra installed, an agent whose input holds the URL of
a CSV, TSV or Excel file reads it with tools that compute over every row,
instead of pulling the file into its context. The URLs come only from the
agent's own input; a URL a model or tool produces is never fetched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib.util import find_spec
import logging
from pathlib import PurePosixPath
import re
from typing import TYPE_CHECKING, Final

from crewai.utilities.message_utils import message_content_text


if TYPE_CHECKING:
    from crewai_files import FileInput

    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage


logger = logging.getLogger(__name__)

# Up to whitespace or a character that closes the prose or markup around a URL;
# a presigned URL percent-encodes all of these.
_URL: Final = re.compile(r"https?://[^\s<>\"'`()\[\]{}|\\^]+")
_TRAILING: Final = ".,;:!?"
_UNREADABLE: Final = (
    "can't be read: it could not be downloaded from its URL (it may be over "
    "the {limit} MB limit, expired, or at an address that can't be fetched)."
)


@dataclass(frozen=True)
class SpreadsheetInputs:
    """The overview block for an agent's prompt and the tools that answer about
    its spreadsheets; empty when its input names none."""

    overview: str = ""
    tools: list[BaseTool] = field(default_factory=list)


def spreadsheets_installed() -> bool:
    """Whether the ``spreadsheets`` extra is installed: the engine's pandas and
    python-calamine, and crewai-tools for its SSRF-guarded download."""
    return all(
        find_spec(module) is not None
        for module in ("crewai_files", "pandas", "python_calamine", "crewai_tools")
    )


def kickoff_input_text(messages: str | list[LLMMessage]) -> str:
    """The text of a kickoff's user input; an assistant or tool message is the
    model's, so a URL in one is not the agent's input."""
    if isinstance(messages, str):
        return messages
    return "\n".join(
        message_content_text(message)
        for message in messages
        if message.get("role") == "user"
    )


def spreadsheet_urls(text: str) -> dict[str, str]:
    """Each spreadsheet URL in ``text``, by the name it is given (its file name,
    numbered when two share one), in order of appearance. Nothing is
    downloaded to decide."""
    from crewai_files import File, FileUrl
    from crewai_files.spreadsheets import is_spreadsheet, url_filename

    found: dict[str, str] = {}
    for match in _URL.finditer(text):
        url = match.group().rstrip(_TRAILING)
        if url in found.values() or not is_spreadsheet(File(source=FileUrl(url=url))):
            continue
        stem = url_filename(url) or "spreadsheet"
        name, n = stem, 1
        while name in found:
            n += 1
            path = PurePosixPath(stem)
            name = f"{path.stem} ({n}){path.suffix}"
        found[name] = url
    return found


def read_spreadsheet_inputs(text: str) -> SpreadsheetInputs:
    """Download and read the spreadsheets whose URLs ``text`` holds.

    Blocks on the network and on parsing, so async callers run it in a thread.
    Returns nothing without the ``spreadsheets`` extra or a spreadsheet URL.
    """
    if not spreadsheets_installed():
        return SpreadsheetInputs()
    urls = spreadsheet_urls(text)
    if not urls:
        return SpreadsheetInputs()

    from crewai_files import File, FileBytes

    from crewai.tools.agent_tools.spreadsheet_tools import (
        spreadsheet_overviews,
        spreadsheet_tools,
    )

    files: dict[str, FileInput] = {}
    unread: dict[str, str] = {}
    for name, url in urls.items():
        data = _download(name, url)
        if data is None:
            unread[name] = _UNREADABLE.format(limit=_max_megabytes())
        else:
            files[name] = File(source=FileBytes(data=data, filename=name))
    tools = (
        spreadsheet_tools(
            files, {url: name for name, url in urls.items() if name in files}
        )
        if files
        else []
    )
    return SpreadsheetInputs(spreadsheet_overviews(files, unread), tools)


def with_spreadsheet_tools(
    tools: list[BaseTool], spreadsheets: SpreadsheetInputs
) -> list[BaseTool]:
    """``tools`` plus the spreadsheet tools, unless the agent already has a tool
    of the same name, which is kept."""
    from crewai.utilities.string_utils import sanitize_tool_name

    taken = {sanitize_tool_name(tool.name) for tool in tools}
    return [
        *tools,
        *(t for t in spreadsheets.tools if sanitize_tool_name(t.name) not in taken),
    ]


def _download(name: str, url: str) -> bytes | None:
    """The file at ``url``, through the guard URLReadTool uses: private and
    reserved addresses are refused on every redirect, and the body is capped.

    None when it can't be had. The reason is not passed on: the guard's
    messages carry the URL, whose query holds a signature.
    """
    from crewai_files.spreadsheets.reading import MAX_BYTES
    from crewai_tools.security.safe_requests import safe_get_bounded
    import requests

    try:
        body, _, _ = safe_get_bounded(url, max_bytes=MAX_BYTES)
    # ValueError: a refused address, too many redirects, or over the cap.
    except (ValueError, requests.RequestException) as e:
        logger.warning(
            "Could not download the spreadsheet %s (%s)", name, type(e).__name__
        )
        return None
    return body


def _max_megabytes() -> int:
    from crewai_files.spreadsheets.reading import MAX_BYTES

    return MAX_BYTES // 1_048_576
