"""Read a spreadsheet's bytes (.xlsx, CSV or TSV) into typed tables.

The bytes come from whoever supplied the file, so every limit is checked before a
parser allocates for it: the archive is inflated and counted first, and each
sheet's extent is measured from its XML before calamine reads it.
"""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass, field
import datetime
import functools
import io
import math
import re
import string
import struct
import threading
from typing import IO, Any, Final, TypeAlias
from xml.parsers import expat
import zipfile
import zlib

import charset_normalizer
import numpy as np
import pandas as pd


MAX_BYTES: Final = 50 * 1024 * 1024
_MAX_MEMBERS: Final = 10_000
# Checked on the inflated bytes, not the sizes the archive declares.
_MAX_UNPACKED: Final = 300_000_000
# Cells from A1 to each sheet's farthest value, summed over the workbook. calamine
# fills a sheet's whole rectangle before any of it reaches Python, so one value at
# XFD1048576 would make a 5 KB file 17 billion cells.
_MAX_CELLS: Final = 10_000_000
_MAX_SHEETS: Final = 50
_MAX_COLUMNS: Final = 2_000
_HEADER_SCAN: Final = 20
_SNIFF_BYTES: Final = 64 * 1024
# Each parse of a large workbook holds hundreds of MB; more than two at once waits.
_PARSE_SLOTS: Final = 2
_PARSE_WAIT_SECONDS: Final = 120.0

_ZIP_MAGIC: Final = b"PK\x03\x04"
# An OLE compound file: an encrypted .xlsx, or a legacy .xls. calamine's parser for
# these has a known denial of service (RUSTSEC-2026-0436), so they never reach it.
_CFB_MAGIC: Final = b"\xd0\xcf\x11\xe0"
_ODS_MIMETYPE: Final = b"application/vnd.oasis.opendocument.spreadsheet"

# Microseconds rather than pandas' default nanoseconds, whose range ends in 2262:
# "valid until 9999-12-31" is a date like any other.
_DATES: Final = "datetime64[us]"

_US_NUMBER: Final = re.compile(r"[-+]?(\d{1,3}(,\d{3})+|\d+)(\.\d+)?([eE][-+]?\d+)?")
_EU_NUMBER: Final = re.compile(r"[-+]?(\d{1,3}(\.\d{3})+|\d+)(,\d+)?")
_DATE_TEXT: Final = re.compile(
    r"\d{4}-\d{1,2}-\d{1,2}([ T]\d{1,2}:\d{2}(:\d{2})?)?|\d{1,2}[./]\d{1,2}[./]\d{2,4}"
)
_CELL_REFERENCE: Final = re.compile(r"([A-Za-z]*)(\d+)")
# A year pandas' nanosecond dates can't hold (they end in April 2262).
_LATE_YEAR: Final = r"(?<!\d)(?:226[2-9]|22[7-9]\d|2[3-9]\d\d|[3-9]\d{3})(?!\d)"
ABOVE_CELL: Final = 40

_parses = threading.BoundedSemaphore(_PARSE_SLOTS)

# A table column holds floats, datetime64 or objects (text, booleans) depending on
# the type inferred for that column while reading, so pandas-stubs can't be given
# one element type for it.
Column: TypeAlias = "pd.Series[Any]"

Cell = (
    str | int | float | bool | datetime.date | datetime.time | datetime.timedelta | None
)


class SpreadsheetError(Exception):
    """A file that can't be read as a spreadsheet; the message says why."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class QueryError(Exception):
    """A request about a spreadsheet that can be fixed; the message says how."""


@dataclass
class Table:
    """A sheet read as a table under one header row."""

    sheet: str
    header_row: int  # 1-based
    detected: bool
    above: list[list[str]]
    frame: pd.DataFrame  # one typed column per header cell; index = 1-based sheet row
    kinds: dict[str, str]
    skipped: dict[str, int]  # filled cells that are not of the column's type
    last_raw: list[Cell]  # the last data row as read, before typing
    decimal_comma: bool
    more_columns: int = 0


@dataclass
class Sheet:
    """One sheet's cells, from A1, with trailing empty rows and columns trimmed."""

    name: str
    hidden: bool
    grid: list[list[Cell]]
    decimal_comma: bool = False
    more_columns: int = 0
    _detected: Table | None = None
    # One other header row at a time, so asking for many can't multiply memory.
    _chosen: Table | None = None

    def table(self, header_row: int = 0) -> Table:
        """The sheet as a table under ``header_row`` (1-based), or the detected one."""
        if not header_row:
            if self._detected is None:
                self._detected = _build_table(self, _detect_header(self.grid), True)
            return self._detected
        if not 1 <= header_row <= max(len(self.grid), 1):
            raise QueryError(
                f"header_row must be 0 (detected) or 1 to {len(self.grid)} for sheet "
                f'"{self.name}"; got {header_row}.'
            )
        # Return the table built here: another caller may replace _chosen meanwhile.
        chosen = self._chosen
        if chosen is None or chosen.header_row != header_row:
            chosen = _build_table(self, header_row - 1, False)
            self._chosen = chosen
        return chosen


@dataclass
class Workbook:
    """The sheets read from one file."""

    kind: str  # "xlsx", "csv" or "tsv"
    sheets: list[Sheet]
    more_sheets: int = 0
    # Sheets past _MAX_SHEETS, and sheets left unread because their extent would
    # pass _MAX_CELLS (None when it couldn't be measured).
    unread_sheets: list[str] = field(default_factory=list)
    too_far: dict[str, tuple[int, int] | None] = field(default_factory=dict)


_PROTECTED_OR_XLS: Final = (
    "the file is password-protected or a legacy .xls workbook, so it can't be read; "
    "an unprotected .xlsx or a CSV can"
)
_TOO_FAR: Final = (
    "its cells reach so far past the data (a value far below or to the right of it) "
    "that it is too large to read; clearing the cells past the data, or saving it as "
    "a CSV, fixes that"
)


def parse(data: bytes, filename: str, *, max_bytes: int = MAX_BYTES) -> Workbook:
    """Read ``data`` as a workbook, by its bytes rather than its name.

    Args:
        data: The file's content.
        filename: The file's name; ``.csv`` and ``.tsv`` mark delimited text.
        max_bytes: The largest file accepted.

    Raises:
        SpreadsheetError: The file is too large, protected, not a spreadsheet, or
            past a limit; the message says which.
    """
    if len(data) > max_bytes:
        raise SpreadsheetError(
            "too_large",
            f"the file is {len(data) / 1_048_576:.1f} MB, over the "
            f"{max_bytes / 1_048_576:.0f} MB limit for spreadsheets",
        )
    if data.startswith(_CFB_MAGIC):
        raise SpreadsheetError("password_or_xls", _PROTECTED_OR_XLS)
    if data.startswith(_ZIP_MAGIC):
        return _read_xlsx(data)
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension in ("csv", "tsv"):
        return _read_text(data, extension)
    raise SpreadsheetError(
        "not_a_spreadsheet", "the file is not an .xlsx, CSV or TSV spreadsheet"
    )


# ---- .xlsx ----------------------------------------------------------------------


def _read_xlsx(data: bytes) -> Workbook:
    from python_calamine import (
        CalamineWorkbook,
        PasswordError,
        SheetTypeEnum,
        SheetVisibleEnum,
    )

    # Which sheets fit is decided from their XML alone, so a sheet that doesn't is
    # never handed to calamine, and calamine never opens a file where none does.
    extents = _measure_archive(data)
    readable: set[str] = set()
    too_far: dict[str, tuple[int, int] | None] = {}
    cells_left = _MAX_CELLS
    for sheet_name, extent in list(extents.items())[:_MAX_SHEETS]:
        if extent is None or extent[0] * extent[1] > cells_left:
            too_far[" ".join(sheet_name.split())] = extent
            continue
        cells_left -= extent[0] * extent[1]
        readable.add(sheet_name)
    unread = [" ".join(n.split()) for n in list(extents)[_MAX_SHEETS:]]
    if not readable:
        if too_far:
            raise SpreadsheetError("too_far", _TOO_FAR)
        return Workbook("xlsx", [], more_sheets=len(unread), unread_sheets=unread)
    _take_a_parse_slot()
    sheets: list[Sheet] = []
    try:
        book = CalamineWorkbook.from_filelike(io.BytesIO(data))
        for meta in book.sheets_metadata:
            if meta.typ != SheetTypeEnum.WorkSheet or meta.name not in readable:
                continue
            # Formulas read as their cached value; one saved without it is empty.
            cells: list[list[Cell]] = [
                list(row)
                for row in book.get_sheet_by_name(meta.name).to_python(
                    skip_empty_area=False
                )
            ]
            grid, more = _trim(cells)
            sheets.append(
                Sheet(
                    name=" ".join(meta.name.split()),
                    hidden=meta.visible != SheetVisibleEnum.Visible,
                    grid=grid,
                    more_columns=more,
                )
            )
    except PasswordError as e:
        raise SpreadsheetError("password_or_xls", _PROTECTED_OR_XLS) from e
    # Any other failure to read a stranger's bytes is the same answer.
    except Exception as e:
        raise _unreadable() from e
    except BaseException as e:
        # A panic in calamine's Rust surfaces as pyo3's PanicException, which derives
        # from BaseException and can't be imported by name; anything else goes on.
        if (type(e).__module__, type(e).__name__) != ("pyo3_runtime", "PanicException"):
            raise
        raise _unreadable() from e
    finally:
        _parses.release()
    return Workbook(
        "xlsx", sheets, more_sheets=len(unread), unread_sheets=unread, too_far=too_far
    )


def _take_a_parse_slot() -> None:
    """Wait for one of the few parse slots, so concurrent uploads can't each
    allocate a full workbook at once; release it with ``_parses.release()``."""
    if not _parses.acquire(timeout=_PARSE_WAIT_SECONDS):
        raise SpreadsheetError(
            "busy", "too many spreadsheets are being read at once; try again"
        )


def _measure_archive(data: bytes) -> dict[str, tuple[int, int] | None]:
    """Each worksheet's extent (rows, columns) from A1 to its farthest value, by
    sheet name in workbook order, after refusing a zip bomb. None for a sheet whose
    cells couldn't be found, which is not read."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if (
                len(members) > _MAX_MEMBERS
                or sum(m.file_size for m in members) > _MAX_UNPACKED
            ):
                raise _too_big_unpacked()
            unpacked = 0
            for member in members:
                if member.flag_bits & 0x1:
                    raise SpreadsheetError("password_or_xls", _PROTECTED_OR_XLS)
                size = _inflated_size(data, member, _MAX_UNPACKED - unpacked)
                if size != member.file_size:
                    raise _unreadable()
                unpacked += size
            parts = _parts(archive)
            if "_rels/.rels" not in parts:
                mimetype = parts.get("mimetype")
                if mimetype is not None:
                    with archive.open(mimetype) as stream:
                        if (
                            stream.read(len(_ODS_MIMETYPE) + 64).strip()
                            == _ODS_MIMETYPE
                        ):
                            raise SpreadsheetError(
                                "ods",
                                "the file is an OpenDocument spreadsheet (.ods), "
                                "which can't be read; an .xlsx or a CSV can",
                            )
                raise _unreadable()
            folder = _workbook_folder(archive.read(parts["_rels/.rels"]))
            extents: dict[str, tuple[int, int] | None] = {}
            for sheet, path in _worksheet_paths(archive, parts, folder).items():
                part = _member(parts, archive, path)
                if part is None:
                    extents[sheet] = None
                    continue
                with archive.open(part) as stream:
                    extents[sheet] = _sheet_extent(stream)
            return extents
    except (zipfile.BadZipFile, zipfile.LargeZipFile, zlib.error, struct.error) as e:
        raise _unreadable() from e
    except expat.ExpatError as e:
        raise _unreadable() from e


def _too_big_unpacked() -> SpreadsheetError:
    return SpreadsheetError(
        "too_big_unpacked",
        f"the workbook unpacks to more than {_MAX_UNPACKED // 1_000_000} MB, too much "
        "to read",
    )


def _unreadable() -> SpreadsheetError:
    return SpreadsheetError("unreadable", "the file could not be read as a spreadsheet")


def _inflated_size(data: bytes, member: zipfile.ZipInfo, budget: int) -> int:
    """The member's real inflated size, counted chunk by chunk against ``budget``,
    so a size the archive declares falsely never matters."""
    offset = member.header_offset
    if data[offset : offset + 4] != _ZIP_MAGIC:
        raise _unreadable()
    name_length, extra_length = struct.unpack("<HH", data[offset + 26 : offset + 30])
    start = offset + 30 + name_length + extra_length
    raw = memoryview(data)[start : start + member.compress_size]
    if member.compress_type == zipfile.ZIP_STORED:
        size = len(raw)
    elif member.compress_type == zipfile.ZIP_DEFLATED:
        inflater, size = zlib.decompressobj(-15), 0
        for at in range(0, len(raw), 1 << 16):
            chunk: bytes = raw[at : at + (1 << 16)].tobytes()
            while chunk:
                size += len(inflater.decompress(chunk, 1 << 20))
                if size > budget:
                    raise _too_big_unpacked()
                chunk = inflater.unconsumed_tail
    else:
        raise _unreadable()
    if size > budget:
        raise _too_big_unpacked()
    return size


# calamine (0.36 and 0.37) finds a part by its name lowercased, ASCII only, with
# backslashes read as slashes, falling back to the exact name. The scan must open
# the same members calamine will, or it measures a decoy.
_ASCII_LOWER: Final = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def _part_key(name: str) -> str:
    return name.replace("\\", "/").translate(_ASCII_LOWER)


def _parts(archive: zipfile.ZipFile) -> dict[str, str]:
    """Each member by the name calamine looks it up under, refusing an archive
    where two members share one: calamine would read the later while this could
    measure the other."""
    parts: dict[str, str] = {}
    for name in archive.namelist():
        key = _part_key(name)
        if key in parts:
            raise _unreadable()
        parts[key] = name
    return parts


def _member(parts: dict[str, str], archive: zipfile.ZipFile, path: str) -> str | None:
    found = parts.get(path.translate(_ASCII_LOWER))
    if found is not None:
        return found
    return path if path in archive.NameToInfo else None


def _target(value: str) -> str:
    """A relationship target as a part path. One with XML-special characters is
    refused, since not every calamine version unescapes them the way expat does."""
    if any(c in value for c in "&<>\"'"):
        raise _unreadable()
    return value


def _workbook_folder(package_rels: bytes) -> str:
    """The folder calamine reads ``workbook.xml`` from: that of the last
    ``officeDocument`` target in ``_rels/.rels``, whatever file it names."""
    targets: list[str] = []

    def relationship(name: str, attributes: dict[str, str]) -> None:
        if (
            name.rsplit("}", 1)[-1] == "Relationship"
            and attributes.get("Type", "").endswith("/relationships/officeDocument")
            and "Target" in attributes
        ):
            targets.append(_target(attributes["Target"]))

    _parse_xml(package_rels, relationship)
    if not targets:
        raise _unreadable()
    return targets[-1][: targets[-1].rfind("/") + 1].removeprefix("/")


def _worksheet_paths(
    archive: zipfile.ZipFile, parts: dict[str, str], folder: str
) -> dict[str, str]:
    """Each worksheet's name and the part path calamine opens for it, from
    ``workbook.xml`` and its relationships in ``folder``. Two sheets with one name
    are refused: calamine opens the first, this would measure the last."""
    rels = _member(parts, archive, f"{folder}_rels/workbook.xml.rels")
    workbook = _member(parts, archive, f"{folder}workbook.xml")
    if rels is None or workbook is None:
        raise _unreadable()
    relationships: dict[str, tuple[str, str]] = {}
    paths: dict[str, str] = {}
    titles: set[str] = set()

    def relationship(name: str, attributes: dict[str, str]) -> None:
        if name.rsplit("}", 1)[-1] == "Relationship" and "Id" in attributes:
            relationships[attributes["Id"]] = (
                _target(attributes.get("Target", "")),
                attributes.get("Type", ""),
            )

    def sheet(name: str, attributes: dict[str, str]) -> None:
        if name.rsplit("}", 1)[-1] != "sheet":
            return
        rel_id = ""
        for key, value in attributes.items():
            if key.rsplit("}", 1)[-1] == "id":
                rel_id = value
        title = attributes.get("name", "")
        if rel_id not in relationships or title in titles:
            raise _unreadable()
        titles.add(title)
        target, kind = relationships[rel_id]
        if kind.rsplit("/", 1)[-1] == "worksheet":
            paths[title] = target[1:] if target.startswith("/") else folder + target

    _parse_xml(archive.read(rels), relationship)
    _parse_xml(archive.read(workbook), sheet)
    return paths


def _parse_xml(xml: bytes, on_start: Callable[[str, dict[str, str]], None]) -> None:
    parser = _strict_parser()
    parser.StartElementHandler = on_start
    parser.Parse(xml, True)


def _sheet_extent(stream: IO[bytes]) -> tuple[int, int]:
    """The rows and columns from A1 to the farthest cell holding a value.

    Mirrors where calamine places a cell: at its ``r`` reference when it has one,
    else after the previous cell in the row; a row without ``r`` follows the one
    before, and calamine treats ``<c/>`` and ``<row/>`` as an opened and closed
    element too. Any cell with a child element counts, which can only overstate
    calamine's extent, never understate it.
    """
    state = {"rows": 0, "columns": 0, "row": 0, "column": 0, "in_cell": 0}
    at = [0, 0]
    filled = [False]

    def start(name: str, attributes: dict[str, str]) -> None:
        tag = name.rsplit("}", 1)[-1]
        if state["in_cell"]:
            filled[0] = True
            return
        if tag == "row":
            reference = attributes.get("r")
            if reference is not None:
                state["row"] = _row_index(reference)
        elif tag == "c":
            reference = attributes.get("r")
            if reference is not None:
                at[0], at[1] = _cell_index(reference)
                state["column"] = at[1]
            else:
                at[0], at[1] = state["row"], state["column"]
            state["in_cell"], filled[0] = 1, False

    def end(name: str) -> None:
        tag = name.rsplit("}", 1)[-1]
        if tag == "c" and state["in_cell"]:
            if filled[0]:
                state["rows"] = max(state["rows"], at[0] + 1)
                state["columns"] = max(state["columns"], at[1] + 1)
            state["column"] += 1
            state["in_cell"] = 0
        elif tag == "row" and not state["in_cell"]:
            state["row"] += 1
            state["column"] = 0

    parser = _strict_parser()
    parser.StartElementHandler = start
    parser.EndElementHandler = end
    while chunk := stream.read(1 << 16):
        parser.Parse(chunk, False)
    parser.Parse(b"", True)
    return state["rows"], state["columns"]


def _strict_parser() -> expat.XMLParserType:
    """An expat parser that refuses any DTD: a workbook never declares one, and one
    here would be an entity-expansion attempt."""

    def refuse(*_: object) -> None:
        raise _unreadable()

    parser = expat.ParserCreate(namespace_separator="}")
    parser.StartDoctypeDeclHandler = refuse
    parser.EntityDeclHandler = refuse
    return parser


def _row_index(reference: str) -> int:
    match = _CELL_REFERENCE.fullmatch(reference.strip())
    if match is None or int(match.group(2)) < 1:
        raise _unreadable()
    return int(match.group(2)) - 1


def _cell_index(reference: str) -> tuple[int, int]:
    match = _CELL_REFERENCE.fullmatch(reference.strip())
    if match is None or not match.group(1) or int(match.group(2)) < 1:
        raise _unreadable()
    column = 0
    for letter in match.group(1).upper():
        column = column * 26 + ord(letter) - 64
    return int(match.group(2)) - 1, column - 1


# ---- CSV and TSV ----------------------------------------------------------------


def _read_text(data: bytes, extension: str) -> Workbook:
    # Ten million cells of text take hundreds of MB, as an .xlsx does.
    _take_a_parse_slot()
    try:
        return _parse_text(data, extension)
    finally:
        _parses.release()


def _parse_text(data: bytes, extension: str) -> Workbook:
    text = _decode(data)
    delimiter = "\t" if extension == "tsv" else _sniff_delimiter(text[:_SNIFF_BYTES])
    rows: list[list[Cell]] = []
    cells = 0
    try:
        for row in csv.reader(io.StringIO(text, newline=""), delimiter=delimiter):
            cells += max(len(row), 1)
            if cells > _MAX_CELLS:
                raise SpreadsheetError(
                    "too_big",
                    f"the file has more than {_MAX_CELLS:,} cells, too many to read",
                )
            rows.append(list(row))
    except csv.Error as e:
        raise _unreadable() from e
    grid, more = _trim(rows)
    sheet = Sheet(
        name="data",
        hidden=False,
        grid=grid,
        # A ";" file is a European export, whose numbers read 1.234,56.
        decimal_comma=delimiter == ";",
        more_columns=more,
    )
    return Workbook(extension, [sheet])


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    match = charset_normalizer.from_bytes(data[:_SNIFF_BYTES]).best()
    if match is not None:
        try:
            return data.decode(match.encoding)
        except (UnicodeDecodeError, LookupError):
            pass
    return data.decode("cp1252", errors="replace")


def _sniff_delimiter(sample: str) -> str:
    """The candidate that splits the most lines into the same, largest number of
    fields, quotes respected. A ";" export's 1.234,56 amounts split by "," into
    fewer fields per line."""
    lines = sample.splitlines()[:-1] or sample.splitlines()
    best, best_score = ",", 0
    for candidate in (",", ";", "\t", "|"):
        widths = [len(r) for r in csv.reader(lines, delimiter=candidate) if len(r) > 1]
        if not widths:
            continue
        modal = max(set(widths), key=widths.count)
        score = widths.count(modal) * modal
        if score > best_score:
            best, best_score = candidate, score
    return best


def _blank(value: Cell) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def number_text(value: float) -> str:
    """Whole numbers exact, others to the cent (four significant digits below 1),
    so a total reads the way the sheet's own cells do."""
    if not math.isfinite(value):
        return str(value)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def text_of(value: object, limit: int | None = None) -> str:
    """One line of cell text: whitespace collapsed, so no value starts a new line in
    a prompt or breaks a table row."""
    if value is None or value is pd.NaT:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (int, float)):
        return number_text(float(value))
    if isinstance(value, np.datetime64):
        value = value.astype("datetime64[us]").item()
    if isinstance(value, datetime.datetime):
        midnight = (value.hour, value.minute, value.second) == (0, 0, 0)
        return value.strftime("%Y-%m-%d" if midnight else "%Y-%m-%d %H:%M")
    if isinstance(value, datetime.date):
        return value.strftime("%Y-%m-%d")
    text = " ".join(str(value).split())
    if limit is not None and len(text) > limit:
        text = text[: limit - 1] + "…"
    return text


def _trim(rows: list[list[Cell]]) -> tuple[list[list[Cell]], int]:
    """Rows without the empty ones at the end and the empty columns at the right,
    and how many columns past the cap were dropped."""
    while rows and all(_blank(c) for c in rows[-1]):
        rows.pop()
    width = max(
        (
            next((i + 1 for i in range(len(r) - 1, -1, -1) if not _blank(r[i])), 0)
            for r in rows
        ),
        default=0,
    )
    kept = min(width, _MAX_COLUMNS)
    return [list(r[:kept]) for r in rows], width - kept


# ---- tables ---------------------------------------------------------------------


def _texty(value: Cell) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    stripped = value.strip()
    return not (
        _US_NUMBER.fullmatch(stripped)
        or _EU_NUMBER.fullmatch(stripped)
        or _DATE_TEXT.fullmatch(stripped)
    )


def _detect_header(grid: list[list[Cell]]) -> int:
    """0-based: the first of the first rows that is at least half filled and mostly
    text, else the first row with anything in it."""
    width = max((len(r) for r in grid), default=0)
    for index, row in enumerate(grid[:_HEADER_SCAN]):
        filled = [c for c in row if not _blank(c)]
        if (
            filled
            and len(filled) >= width / 2
            and sum(map(_texty, filled)) > len(filled) / 2
        ):
            return index
    return next((i for i, r in enumerate(grid) if any(not _blank(c) for c in r)), 0)


def _header_names(row: list[Cell], width: int) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for i in range(width):
        cell = row[i] if i < len(row) else None
        name = "" if _blank(cell) else text_of(cell)[:80]
        name = name or f"column_{i + 1}"
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return names


def _build_table(sheet: Sheet, index: int, detected: bool) -> Table:
    width = max((len(r) for r in sheet.grid), default=0)
    names = _header_names(sheet.grid[index] if sheet.grid else [], width)
    # Inline rather than a helper per cell: this runs once for every cell.
    body = [
        [(cell.strip() or None) if isinstance(cell, str) else cell for cell in row]
        + [None] * (width - len(row))
        for row in sheet.grid[index + 1 :]
    ]
    raw = pd.DataFrame(body, dtype=object, columns=range(width))
    raw.index = pd.RangeIndex(index + 2, index + 2 + len(body))
    if width:
        raw = raw[raw.notna().any(axis=1)]
    columns: dict[str, Column] = {}
    kinds: dict[str, str] = {}
    skipped: dict[str, int] = {}
    for position, name in enumerate(names):
        columns[name], kinds[name], skipped[name] = _typed(
            raw[position], sheet.decimal_comma
        )
    frame = pd.DataFrame(columns, index=raw.index)
    above = [
        [text_of(c)[:ABOVE_CELL] for c in row if not _blank(c)]
        for row in sheet.grid[:index]
    ]
    return Table(
        sheet=sheet.name,
        header_row=index + 1,
        detected=detected,
        above=above,
        frame=frame,
        kinds=kinds,
        skipped=skipped,
        last_raw=list(raw.iloc[-1]) if len(raw) else [],
        decimal_comma=sheet.decimal_comma,
        more_columns=sheet.more_columns,
    )


def numbers(texts: Column, decimal_comma: bool) -> Column:
    """Text that reads as a number, as floats; NaN where it doesn't."""
    cleaned = texts.str.replace("\N{NO-BREAK SPACE}", "", regex=False).str.replace(
        " ", "", regex=False
    )
    if decimal_comma:
        matched = cleaned.str.fullmatch(_EU_NUMBER.pattern)
        cleaned = cleaned.str.replace(".", "", regex=False).str.replace(
            ",", ".", regex=False
        )
    else:
        matched = cleaned.str.fullmatch(_US_NUMBER.pattern)
        cleaned = cleaned.str.replace(",", "", regex=False)
    keep = matched.fillna(False).astype(bool)
    return pd.to_numeric(cleaned.where(keep), errors="coerce")


def _dates(texts: Column, dayfirst: bool) -> Column:
    """Text that reads as a date under the one format most of it fits. Both slash
    orders are tried; on a tie the file's convention wins."""
    slashes = ("%d/%m/%Y", "%m/%d/%Y") if dayfirst else ("%m/%d/%Y", "%d/%m/%Y")
    best = pd.Series(pd.NaT, index=texts.index, dtype=_DATES)
    for form in ("ISO8601", "%d.%m.%Y", *slashes, "%d.%m.%y"):
        parsed = pd.to_datetime(texts, errors="coerce", format=form).astype(_DATES)
        # pandas 2 parses to nanoseconds, whose range ends in 2262; Python places
        # the dates it couldn't.
        missed = texts[parsed.isna()]
        missed = missed[missed.str.contains(_LATE_YEAR, regex=True)]
        if len(missed):
            late = missed.map(functools.partial(date_from_text, form=form)).dropna()
            if len(late):
                parsed[late.index] = late.astype(_DATES)
        if parsed.notna().sum() > best.notna().sum():
            best = parsed
    return best


def date_from_text(text: str, form: str) -> datetime.datetime | None:
    """``text`` as a date in ``form`` (or "ISO8601"), with Python's parser, whose
    range runs to the year 9999; None when it doesn't read."""
    try:
        if form == "ISO8601":
            return datetime.datetime.fromisoformat(text)
        return datetime.datetime.strptime(text, form)
    except ValueError:
        return None


def _cell_kind(value: Cell) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "text"
    return "date" if isinstance(value, datetime.date) else "other"


def _as_datetime(value: datetime.date) -> datetime.datetime:
    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime(value.year, value.month, value.day)


def _typed(raw: Column, decimal_comma: bool) -> tuple[Column, str, int]:
    """``raw`` (stripped text, None for blank) as its type: integer, number, date,
    boolean, text or mixed (no type covers 90% of the filled cells), and how many
    filled cells don't fit that type."""
    filled = raw.notna()
    count = int(filled.sum())
    if not count:
        return pd.Series(None, index=raw.index, dtype=object), "empty", 0
    present = raw[filled]
    cell_kinds = present.map(_cell_kind)
    is_bool, is_number = cell_kinds == "boolean", cell_kinds == "number"
    is_date, is_text = cell_kinds == "date", cell_kinds == "text"
    texts = present[is_text].astype(str)
    numeric_text = (
        numbers(texts, decimal_comma) if len(texts) else pd.Series(dtype=float)
    )
    date_text = (
        _dates(texts[numeric_text.isna()], dayfirst=decimal_comma)
        if len(texts) and texts.str.fullmatch(_DATE_TEXT.pattern).mean() >= 0.5
        else pd.Series(dtype=_DATES)
    )
    tally = {
        "number": int(is_number.sum() + numeric_text.notna().sum()),
        "date": int(is_date.sum() + date_text.notna().sum()),
        "boolean": int(is_bool.sum()),
    }
    kind, hits = max(tally.items(), key=lambda item: item[1])
    if hits >= 0.9 * count:
        if kind == "number":
            out = pd.Series(float("nan"), index=raw.index, dtype=float)
            native = present[is_number].astype(float)
            out[native.index] = native
            out[numeric_text.dropna().index] = numeric_text.dropna()
            finite = out.dropna()
            if len(finite) and bool((finite == finite.round()).all()):
                kind = "integer"
        elif kind == "date":
            out = pd.Series(pd.NaT, index=raw.index, dtype=_DATES)
            moments = present[is_date].map(_as_datetime).astype(_DATES)
            out[moments.index] = moments
            out[date_text.dropna().index] = date_text.dropna()
        else:
            out = pd.Series(None, index=raw.index, dtype=object)
            out[present[is_bool].index] = present[is_bool]
        return out, kind, count - int(out.notna().sum())
    out = pd.Series(None, index=raw.index, dtype=object)
    out[present.index] = present.map(text_of)
    return out, "text" if count - sum(tally.values()) >= 0.9 * count else "mixed", 0
