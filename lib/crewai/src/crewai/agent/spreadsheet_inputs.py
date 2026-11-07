"""Spreadsheets an agent is given as URLs in its input.

With the ``spreadsheets`` extra installed, an agent whose input holds the URL of
a CSV, TSV or Excel file reads it with tools that compute over every row,
instead of pulling the file into its context.

Only URLs from input a caller gave are fetched, never one a model wrote. A run's
own input (a kickoff's user messages, a task's interpolated description) is
trusted; an agent run inside it (a coworker given a delegated task) may read
only the URLs the outer run's input held, and the runs crewAI makes on model
text (a guardrail check, crew planning, evaluation) read none.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
import html
from importlib.util import find_spec
import logging
from pathlib import PurePosixPath
import re
import threading
import time
from typing import TYPE_CHECKING, Final

from crewai.utilities.message_utils import message_content_text


if TYPE_CHECKING:
    from crewai_files import FileInput

    from crewai.task import Task
    from crewai.tools.base_tool import BaseTool
    from crewai.utilities.types import LLMMessage


logger = logging.getLogger(__name__)

# ASCII URL characters only: prose around a URL (quotes, brackets, CJK
# punctuation) ends it, and urlparse never sees text it rejects.
_URL: Final = re.compile(r"https?://[A-Za-z0-9\-._~:/?#@!$&*+,;=%()]+")
# Markup escapes a URL's "&" as "&amp;". Only references ending in ";" are
# decoded: html.unescape alone also reads "&para=2" in a query as "¶=2".
_CHARACTER_REFERENCE: Final = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z]+);")
_TRAILING: Final = ".,;:!?"
_MAX_URLS: Final = 10
_UNREADABLE: Final = (
    "can't be read: it could not be downloaded from its URL (it may be over "
    "the {limit} MB limit, too slow to download, expired, or at an address "
    "that can't be fetched)."
)
_WEB_PAGE: Final = (
    "can't be read: its URL serves a web page, not the file (a share or preview "
    "link); a direct download link works."
)
_OVER_LIMIT: Final = (
    f"  - {{count}} more spreadsheet URLs were not read: only the first "
    f"{_MAX_URLS} are."
)
# For all of one run's downloads together: they happen before the run starts,
# outside its max_execution_time.
_DOWNLOAD_SECONDS: Final = 60.0
# A retry, a guardrail retry or the next turn of a conversation reads the same
# URLs again within minutes; a URL's content can change, so not for longer.
_DOWNLOAD_TTL_SECONDS: Final = 300.0
_CACHED_DOWNLOADS: Final = 8

# While an agent runs, the spreadsheet URLs its own input gave it; None outside
# any run.
_trusted_urls: ContextVar[frozenset[str] | None] = ContextVar(
    "crewai_trusted_spreadsheet_urls", default=None
)
_downloads: OrderedDict[str, tuple[float, bytes]] = OrderedDict()
_downloads_lock = threading.Lock()


@dataclass(frozen=True)
class SpreadsheetInputs:
    """The overview block for an agent's prompt, the tools that answer about
    its spreadsheets, and the URLs it read; empty when its input names none."""

    overview: str = ""
    tools: list[BaseTool] = field(default_factory=list)
    urls: frozenset[str] = frozenset()


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


def task_input_text(task: Task) -> str:
    """A task's description as its inputs filled it in, before crew planning or
    a crew chat's history is added to it, which are model text."""
    return task._input_description or task.description


def spreadsheet_urls(text: str) -> dict[str, str]:
    """Each spreadsheet URL in ``text``, by the name it is given (its file name,
    numbered when two share one), in order of appearance. Nothing is
    downloaded to decide."""
    from crewai_files import File, FileUrl
    from crewai_files.spreadsheets import is_spreadsheet, url_filename

    text = _CHARACTER_REFERENCE.sub(lambda m: html.unescape(m.group()), text)
    found: dict[str, str] = {}
    seen: set[str] = set()
    taken: set[str] = set()
    last_number: dict[str, int] = {}
    for match in _URL.finditer(text):
        url = _trim(match.group())
        if url in seen:
            continue
        seen.add(url)
        if not is_spreadsheet(File(source=FileUrl(url=url))):
            continue
        stem = url_filename(url) or "spreadsheet"
        name = stem
        while name.casefold() in taken:
            n = last_number.get(stem.casefold(), 1) + 1
            last_number[stem.casefold()] = n
            path = PurePosixPath(stem)
            name = f"{path.stem} ({n}){path.suffix}"
        found[name] = url
        taken.add(name.casefold())
    return found


def read_spreadsheet_inputs(text: str) -> SpreadsheetInputs:
    """Download and read the spreadsheets whose URLs ``text`` holds, keeping to
    the URLs an enclosing run trusts.

    Blocks on the network and on parsing, so async callers run it in a thread.
    Returns nothing without the ``spreadsheets`` extra or a spreadsheet URL.
    """
    if not spreadsheets_installed():
        return SpreadsheetInputs()
    urls = spreadsheet_urls(text)
    trusted = _trusted_urls.get()
    if trusted is not None:
        urls = {name: url for name, url in urls.items() if url in trusted}
    if not urls:
        return SpreadsheetInputs()

    from crewai_files import File, FileBytes

    from crewai.tools.agent_tools.spreadsheet_tools import (
        spreadsheet_overviews,
        spreadsheet_tools,
    )

    over = len(urls) - _MAX_URLS
    files: dict[str, FileInput] = {}
    unread: dict[str, str] = {}
    deadline = time.monotonic() + _DOWNLOAD_SECONDS
    for name, url in list(urls.items())[:_MAX_URLS]:
        downloaded = _download(name, url, deadline)
        if isinstance(downloaded, str):
            unread[name] = downloaded
        else:
            files[name] = File(source=FileBytes(data=downloaded, filename=name))
    read = {url: name for name, url in urls.items() if name in files}
    overview = spreadsheet_overviews(files, unread)
    if over > 0:
        overview += "\n" + _OVER_LIMIT.format(count=over)
    return SpreadsheetInputs(
        overview,
        spreadsheet_tools(files, read) if files else [],
        frozenset(read),
    )


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


def enter_spreadsheet_scope(
    spreadsheets: SpreadsheetInputs,
) -> Token[frozenset[str] | None]:
    """Start a run that trusts the URLs its input gave it; an agent run inside
    it reads only those. A run inside another keeps the outer run's trust."""
    outer = _trusted_urls.get()
    return _trusted_urls.set(spreadsheets.urls if outer is None else outer)


def exit_spreadsheet_scope(token: Token[frozenset[str] | None]) -> None:
    _trusted_urls.reset(token)


@contextmanager
def no_spreadsheet_urls() -> Iterator[None]:
    """For a run crewAI makes on model text: it reads no spreadsheet URL."""
    token = _trusted_urls.set(frozenset())
    try:
        yield
    finally:
        _trusted_urls.reset(token)


def _trim(url: str) -> str:
    """``url`` without the punctuation and unbalanced ``)`` that end the
    sentence or markup around it."""
    while True:
        trimmed = url.rstrip(_TRAILING)
        if trimmed.endswith(")") and trimmed.count(")") > trimmed.count("("):
            trimmed = trimmed[:-1]
        if trimmed == url:
            return url
        url = trimmed


def _download(name: str, url: str, deadline: float) -> bytes | str:
    """The file at ``url``, through the guard URLReadTool uses: private and
    reserved addresses are refused on every redirect, and the body is capped
    and abandoned at ``deadline`` (a ``time.monotonic()`` value).

    The reason it can't be read otherwise, without the guard's own message,
    which carries the URL and so its signature.
    """
    from crewai_files.spreadsheets.reading import MAX_BYTES
    from crewai_tools.security.safe_requests import safe_get_bounded
    import requests

    now = time.monotonic()
    with _downloads_lock:
        for stale in [
            key
            for key, (stored, _) in _downloads.items()
            if now - stored >= _DOWNLOAD_TTL_SECONDS
        ]:
            del _downloads[stale]
        cached = _downloads.get(url)
        if cached is not None:
            _downloads.move_to_end(url)
            return cached[1]
    unreadable = _UNREADABLE.format(limit=MAX_BYTES // 1_048_576)
    if now >= deadline:
        return unreadable
    try:
        body, content_type, _ = safe_get_bounded(
            url,
            max_bytes=MAX_BYTES,
            timeout=min(30.0, deadline - now),
            deadline=deadline,
        )
    # ValueError: a refused address, too many redirects, over the cap, or
    # past the deadline.
    except (ValueError, requests.RequestException) as e:
        logger.warning(
            "Could not download the spreadsheet %s (%s)", name, type(e).__name__
        )
        return unreadable
    if content_type.split(";")[0].strip().lower() in (
        "text/html",
        "application/xhtml+xml",
    ):
        return _WEB_PAGE
    with _downloads_lock:
        _downloads[url] = (now, body)
        _downloads.move_to_end(url)
        while len(_downloads) > _CACHED_DOWNLOADS:
            _downloads.popitem(last=False)
    return body
