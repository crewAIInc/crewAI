"""Spreadsheets (.xlsx, CSV, TSV) read in Python, for agents to ask about.

Internal for now: crewAI's agent tools are the only caller, and the API may
change. ``is_spreadsheet`` needs nothing extra; reading needs the
``spreadsheets`` extra (pandas, python-calamine, charset-normalizer), and the
answers live in ``crewai_files.spreadsheets.answers``.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import io
from pathlib import Path, PurePosixPath
import threading
from typing import TYPE_CHECKING, Final
from urllib.parse import urlparse
import zipfile
import zlib

from crewai_files.core.sources import FileBytes, FilePath, FileUrl


if TYPE_CHECKING:
    from crewai_files.core.types import FileInput
    from crewai_files.spreadsheets.reading import Workbook


_EXTENSIONS: Final = frozenset({"csv", "tsv", "xlsx", "xls", "ods"})
_CONTENT_TYPES: Final = frozenset(
    {
        "text/csv",
        "text/tab-separated-values",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.ms-excel",
        "application/vnd.oasis.opendocument.spreadsheet",
    }
)
_ARCHIVE_TYPES: Final = frozenset({"application/zip", "application/octet-stream"})
_DELIMITED: Final = frozenset({"csv", "tsv"})
_DELIMITED_TYPES: Final = {"text/csv": "csv", "text/tab-separated-values": "tsv"}
_CACHED_WORKBOOKS: Final = 8
# Cells held across cached workbooks: one workbook at the parser's own limit,
# about 1 GB in memory, so eight large uploads can't pin several.
_CACHED_CELLS: Final = 10_000_000
# Recognition opens an unknown archive too, so it keeps to the parser's member
# cap and reads no more of ``mimetype`` than the value it compares.
_MAX_MEMBERS: Final = 10_000
_ODS_MIMETYPE: Final = b"application/vnd.oasis.opendocument.spreadsheet"

_workbooks: OrderedDict[tuple[str, str], tuple[Workbook, int]] = OrderedDict()
_workbooks_lock = threading.Lock()


def is_spreadsheet(file: FileInput) -> bool:
    """Whether ``file`` is a spreadsheet, whatever type it arrived as.

    Spreadsheets reach crewAI as ``TextFile``, ``File`` and even ``AudioFile``,
    so this looks at the name, the content type and, for a local file or bytes of
    an unknown type, the archive inside. Legacy ``.xls`` and ``.ods`` count: they
    are spreadsheets the reader refuses with a reason. A URL is never downloaded
    to decide.
    """
    if _extension(file) in _EXTENSIONS:
        return True
    source = file.source
    content_type = file.content_type
    if content_type in _CONTENT_TYPES:
        return True
    if isinstance(source, FilePath) and content_type in _ARCHIVE_TYPES:
        return _holds_a_workbook(source.path)
    if isinstance(source, FileBytes) and content_type in _ARCHIVE_TYPES:
        data = file.read()
        return data.startswith(b"PK\x03\x04") and _holds_a_workbook(io.BytesIO(data))
    return False


def load_spreadsheet(file: FileInput, *, max_bytes: int | None = None) -> Workbook:
    """Read ``file`` as a workbook, reusing the result for the same content.

    A URL is streamed and stopped at ``max_bytes``, so a large file is never held
    whole before it is refused.

    Raises:
        SpreadsheetError: The file can't be read as a spreadsheet; the message
            says why.
    """
    from crewai_files.spreadsheets.reading import MAX_BYTES, parse

    limit = MAX_BYTES if max_bytes is None else max_bytes
    data = _read_capped(file, limit)
    name = file.filename or _url_name(file) or "data"
    if _extension(file) not in _DELIMITED and file.content_type in _DELIMITED_TYPES:
        # Bytes with no name, typed as delimited text by whoever supplied them.
        name = f"{name}.{_DELIMITED_TYPES[file.content_type]}"
    key = (hashlib.sha256(data).hexdigest(), PurePosixPath(name).suffix.lower())
    with _workbooks_lock:
        if key in _workbooks:
            _workbooks.move_to_end(key)
            return _workbooks[key][0]
    workbook = parse(data, name, max_bytes=limit)
    cells = sum(len(row) for sheet in workbook.sheets for row in sheet.grid)
    with _workbooks_lock:
        _workbooks[key] = (workbook, cells)
        while len(_workbooks) > 1 and (
            len(_workbooks) > _CACHED_WORKBOOKS
            or sum(held for _, held in _workbooks.values()) > _CACHED_CELLS
        ):
            _workbooks.popitem(last=False)
    return workbook


def _extension(file: FileInput) -> str:
    name = file.filename or _url_name(file) or ""
    suffix = PurePosixPath(name).suffix
    return suffix[1:].lower() if suffix else ""


def _url_name(file: FileInput) -> str | None:
    source = file.source
    if isinstance(source, FileUrl):
        return PurePosixPath(urlparse(source.url).path).name or None
    return None


def _holds_a_workbook(archive_file: Path | io.BytesIO) -> bool:
    try:
        with zipfile.ZipFile(archive_file) as archive:
            members = archive.infolist()
            if len(members) > _MAX_MEMBERS:
                return False
            if any(m.filename.lower() == "xl/workbook.xml" for m in members):
                return True
            mimetype = next((m for m in members if m.filename == "mimetype"), None)
            if mimetype is None:
                return False
            with archive.open(mimetype) as stream:
                return stream.read(len(_ODS_MIMETYPE) + 64).strip() == _ODS_MIMETYPE
    # A damaged, encrypted or oddly compressed archive isn't one to route here.
    except (
        zipfile.BadZipFile,
        zipfile.LargeZipFile,
        zlib.error,
        NotImplementedError,
        RuntimeError,
        EOFError,
        OSError,
    ):
        return False


def _read_capped(file: FileInput, limit: int) -> bytes:
    from crewai_files.spreadsheets.reading import SpreadsheetError

    source = file.source
    too_large = SpreadsheetError(
        "too_large",
        f"the file is over the {limit / 1_048_576:.0f} MB limit for spreadsheets",
    )
    if isinstance(source, FileUrl):
        return _download(source.url, limit, too_large)
    if isinstance(source, FilePath) and source.path.stat().st_size > limit:
        raise too_large
    data = file.read()
    if len(data) > limit:
        raise too_large
    return data


def _download(url: str, limit: int, too_large: Exception) -> bytes:
    import httpx

    from crewai_files.spreadsheets.reading import SpreadsheetError

    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=30.0) as response:
            response.raise_for_status()
            declared = response.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > limit:
                raise too_large
            body = bytearray()
            for chunk in response.iter_bytes():
                body += chunk
                if len(body) > limit:
                    raise too_large
            return bytes(body)
    except httpx.HTTPError as e:
        raise SpreadsheetError(
            "download", "the file could not be downloaded from its URL"
        ) from e
