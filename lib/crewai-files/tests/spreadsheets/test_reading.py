"""Reading a spreadsheet's bytes: what is refused before a parser allocates for it,
and how cells become typed columns. Every file is built here, in memory.

Ported from Studio Assistant's spreadsheet reader (studio-assistant #370, #374),
with the far-cell guard moved from a child-process probe to a scan of each
sheet's XML, which bounds memory on every OS.
"""

from __future__ import annotations

import datetime
import io
import re
import struct
import threading
import time
import zipfile

from crewai_files.spreadsheets import reading
from crewai_files.spreadsheets.reading import QueryError, SpreadsheetError, parse
import openpyxl
from openpyxl.styles import Font
import pytest


def xlsx(*sheets, far=None) -> bytes:
    """A workbook of ``(name, rows)`` or ``(name, rows, "hidden")`` sheets; ``far``
    puts one value at ``(row, column)`` on the named sheet."""
    book = openpyxl.Workbook()
    book.remove(book.active)
    for name, rows, *hidden in sheets:
        sheet = book.create_sheet(name)
        for row in rows:
            sheet.append(row)
        if hidden:
            sheet.sheet_state = "hidden"
        if far and name in far:
            sheet.cell(row=far[name][0], column=far[name][1], value="x")
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


def rewrite_sheet(data: bytes, change) -> bytes:
    """The workbook with its first sheet's XML passed through ``change``."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts["xl/worksheets/sheet1.xml"] = change(parts["xl/worksheets/sheet1.xml"])
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return out.getvalue()


def with_cells(rows: bytes) -> bytes:
    """A one-sheet workbook whose sheetData is exactly ``rows``."""
    return rewrite_sheet(
        xlsx(("Data", [["id"]])),
        lambda xml: re.sub(
            rb"<sheetData>.*</sheetData>|<sheetData/>",
            lambda _: b"<sheetData>" + rows + b"</sheetData>",
            xml,
            flags=re.S,
        ),
    )


def reason(data: bytes, name: str = "w.xlsx") -> str:
    with pytest.raises(SpreadsheetError) as raised:
        parse(data, name)
    return raised.value.reason


def parse_within(data: bytes, seconds: float = 5.0) -> reading.Workbook:
    started = time.monotonic()
    try:
        return parse(data, "w.xlsx")
    finally:
        assert time.monotonic() - started < seconds


DATA = [["id", "name", "amount"]] + [[i, f"Customer {i}", i * 2.5] for i in range(1, 6)]
KUNDEN = [["Kunde", "Land"], ["Acme", "DE"], ["Brix", "AT"]]


@pytest.fixture
def calamine_must_not_open(monkeypatch):
    """Fails the test if calamine is asked to open the file."""
    import python_calamine

    def refuse(stream):
        raise AssertionError("calamine must never see this file")

    monkeypatch.setattr(python_calamine.CalamineWorkbook, "from_filelike", refuse)


class TestSniffing:
    def test_an_ole_file_is_refused_as_protected_or_legacy(
        self, calamine_must_not_open
    ):
        ole = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + bytes(504)
        with pytest.raises(SpreadsheetError, match="password-protected or a legacy"):
            parse(ole, "Budget.xlsx")

    def test_an_ole_file_named_csv_is_refused_too(self, calamine_must_not_open):
        assert reason(b"\xd0\xcf\x11\xe0" + bytes(100), "export.csv") == (
            "password_or_xls"
        )

    def test_an_xlsx_name_on_other_bytes_is_not_a_spreadsheet(self):
        with pytest.raises(SpreadsheetError, match=r"not an \.xlsx, CSV or TSV"):
            parse(b"%PDF-1.7 not a workbook", "Q3.xlsx")

    def test_a_workbook_named_csv_is_read_as_a_workbook(self):
        book = parse(xlsx(("Kunden", KUNDEN)), "export.csv")
        assert (book.kind, [s.name for s in book.sheets]) == ("xlsx", ["Kunden"])

    def test_a_broken_archive_is_unreadable(self):
        assert reason(b"PK\x03\x04" + bytes(200)) == "unreadable"

    def test_a_zip_that_is_not_a_workbook_is_unreadable(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            archive.writestr("notes.txt", "hello")
        assert reason(out.getvalue()) == "unreadable"

    def test_an_ods_file_says_so(self, calamine_must_not_open):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            archive.writestr(
                "mimetype", "application/vnd.oasis.opendocument.spreadsheet"
            )
            archive.writestr("content.xml", "<office:document-content/>")
        with pytest.raises(
            SpreadsheetError, match=r"OpenDocument spreadsheet \(\.ods\)"
        ):
            parse(out.getvalue(), "budget.ods")

    def test_an_encrypted_member_is_refused_as_protected(self, calamine_must_not_open):
        data = bytearray(xlsx(("Kunden", KUNDEN)))
        # Set the "encrypted" flag on the first member in the central directory.
        at = data.find(b"PK\x01\x02")
        flags = struct.unpack("<H", data[at + 8 : at + 10])[0]
        data[at + 8 : at + 10] = struct.pack("<H", flags | 0x1)
        assert reason(bytes(data)) == "password_or_xls"

    def test_a_file_over_the_byte_limit_is_refused_unread(self, calamine_must_not_open):
        with pytest.raises(SpreadsheetError, match=r"over the 1 MB limit"):
            parse(b"a,b\n" * 300_000, "big.csv", max_bytes=1_048_576)


def _bomb(payload: bytes, declared: int | None = None, members: int = 1) -> bytes:
    """A zip of ``members`` deflated copies of ``payload``; with ``declared``, every
    header claims that uncompressed size instead of the real one."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for i in range(members):
            archive.writestr(f"xl/worksheets/sheet{i}.xml", payload)
    data = bytearray(out.getvalue())
    if declared is not None:
        for signature, size_at in ((b"PK\x03\x04", 22), (b"PK\x01\x02", 24)):
            at = data.find(signature)
            while at != -1:
                data[at + size_at : at + size_at + 4] = struct.pack("<I", declared)
                at = data.find(signature, at + 4)
    return bytes(data)


class TestZipBombs:
    @pytest.fixture(autouse=True)
    def _small_cap(self, monkeypatch, calamine_must_not_open):
        monkeypatch.setattr(reading, "_MAX_UNPACKED", 100_000)

    def test_declared_sizes_over_the_cap_are_refused_unread(self):
        assert reason(_bomb(b"0" * 60_000, members=2)) == "too_big_unpacked"

    def test_a_member_that_inflates_past_its_declared_size_is_refused(self):
        # Declares 10 bytes and holds 50 KB: zipfile would stop at 10 and pass.
        assert reason(_bomb(b"0" * 50_000, declared=10)) == "unreadable"

    def test_a_lying_member_that_inflates_past_the_cap_is_refused(self):
        with pytest.raises(SpreadsheetError, match="unpacks to more than 0 MB"):
            parse(_bomb(b"0" * 500_000, declared=10), "w.xlsx")

    def test_a_member_smaller_than_it_declares_is_refused(self):
        assert reason(_bomb(b"0" * 1_000, declared=5_000)) == "unreadable"

    def test_too_many_members_are_refused(self, monkeypatch):
        monkeypatch.setattr(reading, "_MAX_MEMBERS", 3)
        assert reason(_bomb(b"x", members=4)) == "too_big_unpacked"


class TestFarCells:
    """calamine fills a sheet's rectangle from A1 to its farthest value before any
    of it reaches Python: one value at XFD1048576 in a 5 KB file never finished
    parsing in Studio Assistant, and one at AZ1048576 took 13 s and 2 GB."""

    @pytest.mark.parametrize(
        "corner",
        [(1_048_576, 16_384), (1_048_576, 52), (5_000, 16_384)],
        ids=["XFD1048576", "AZ1048576", "XFD5000"],
    )
    def test_a_value_far_from_the_data_is_refused_at_once(
        self, corner, calamine_must_not_open
    ):
        with pytest.raises(SpreadsheetError, match="cells reach so far past the data"):
            parse_within(xlsx(("Data", DATA), far={"Data": corner}))

    def test_a_stray_value_far_down_one_column_still_reads(self):
        # 1,048,576 rows x 3 columns, inside the 10M-cell budget.
        book = parse_within(xlsx(("Data", DATA), far={"Data": (1_048_576, 1)}), 20.0)
        assert [s.name for s in book.sheets] == ["Data"]

    def test_formatting_far_from_the_data_is_not_a_value(self):
        book = openpyxl.load_workbook(io.BytesIO(xlsx(("Data", DATA))))
        book["Data"].cell(row=1_048_576, column=16_384).font = Font(bold=True)
        out = io.BytesIO()
        book.save(out)
        assert [s.name for s in parse_within(out.getvalue()).sheets] == ["Data"]

    def test_a_dimension_that_claims_the_whole_grid_changes_nothing(self):
        lying = rewrite_sheet(
            xlsx(("Data", DATA)),
            lambda xml: re.sub(
                rb'<dimension ref="[^"]*"/>', b'<dimension ref="A1:XFD1048576"/>', xml
            ),
        )
        assert [s.name for s in parse_within(lying).sheets] == ["Data"]

    def test_cells_and_rows_without_a_position_are_counted_where_they_land(
        self, calamine_must_not_open
    ):
        # calamine places a row or cell with no r="..." after the one before.
        rows = (
            b'<row><c t="inlineStr"><is><t>id</t></is></c></row>'
            + b"<row/>" * 1_048_574
            + b"<row>"
            + b"<c/>" * 16_383
            + b'<c t="n"><v>1</v></c></row>'
        )
        with pytest.raises(SpreadsheetError) as raised:
            parse_within(with_cells(rows), 20.0)
        assert raised.value.reason == "too_far"

    # Ways of writing a position that a regex scan of the XML missed and calamine
    # reads (Studio Assistant #374's security review). An XML parser reads them all.
    @pytest.mark.parametrize(
        "cell",
        [
            b"<c r='XFD1048576' t=\"n\"><v>1</v></c>",
            b'<c r="A99999999" t="n"><v>1</v></c>',
            b'<c x=">" r="XFD1048576" t="n"><v>1</v></c>',
            b'<c r = "XFD1048576" t="n"><v>1</v></c>',
            b'<c r="xfd1048576" t="n"><v>1</v></c>',
        ],
        ids=[
            "single-quotes",
            "row-past-excel",
            "gt-in-attribute",
            "spaced-equals",
            "lowercase",
        ],
    )
    def test_a_position_written_any_way_calamine_reads_is_measured(
        self, cell, calamine_must_not_open
    ):
        rows = (
            b'<row r="1"><c r="A1" t="inlineStr"><is><t>id</t></is></c>'
            + cell
            + b"</row>"
        )
        assert reason(with_cells(rows)) == "too_far"

    def test_a_namespace_prefix_on_cells_is_still_measured(
        self, calamine_must_not_open
    ):
        def prefixed(xml: bytes) -> bytes:
            xml = xml.replace(b"<worksheet ", b"<x:worksheet xmlns:x=" + _MAIN + b" ")
            xml = xml.replace(b"</worksheet>", b"</x:worksheet>")
            return re.sub(
                rb"<sheetData>.*</sheetData>|<sheetData/>",
                b'<x:sheetData><x:row r="1"><x:c r="XFD1048576" t="n"><x:v>1</x:v>'
                b"</x:c></x:row></x:sheetData>",
                xml,
                flags=re.S,
            )

        assert reason(rewrite_sheet(xlsx(("Data", [["id"]])), prefixed)) == "too_far"

    def test_an_unreadable_cell_reference_is_refused(self, calamine_must_not_open):
        assert reason(with_cells(b'<row><c r="A0"><v>1</v></c></row>')) == (
            "unreadable"
        )

    def test_a_far_sheet_costs_only_itself(self):
        book = parse_within(
            xlsx(
                ("Before", DATA),
                ("Far", DATA),
                ("After", DATA),
                far={"Far": (1_048_576, 52)},
            )
        )
        assert [s.name for s in book.sheets] == ["Before", "After"]
        assert book.too_far == {"Far": (1_048_576, 52)}

    def test_the_budget_is_for_the_whole_workbook(self, monkeypatch):
        # Three sheets of 20 rows x 3 columns under a 130-cell budget: two fit.
        monkeypatch.setattr(reading, "_MAX_CELLS", 130)
        rows = [["a", "b", "c"]] + [[i, i, i] for i in range(19)]
        book = parse_within(xlsx(("One", rows), ("Two", rows), ("Three", rows)))
        assert [s.name for s in book.sheets] == ["One", "Two"]
        assert list(book.too_far) == ["Three"]


_MAIN = b'"http://schemas.openxmlformats.org/spreadsheetml/2006/main"'


class TestXmlAttacks:
    @pytest.mark.parametrize(
        "prolog",
        [
            b'<!DOCTYPE x [<!ENTITY a "aaaaaaaa">]>',
            b'<!DOCTYPE x SYSTEM "http://example.invalid/x.dtd">',
        ],
        ids=["entity", "external"],
    )
    def test_a_sheet_that_declares_a_dtd_is_refused(
        self, prolog, calamine_must_not_open
    ):
        def declare(xml: bytes) -> bytes:
            head, _, rest = xml.partition(b"?>")
            return head + b"?>" + prolog + rest

        assert reason(rewrite_sheet(xlsx(("Data", DATA)), declare)) == "unreadable"


class TestWorkbooks:
    BOOK = parse(
        xlsx(
            ("Kunden", KUNDEN),
            ("Lookups", [["k", "v"], ["a", 1]], "hidden"),
        ),
        "w.xlsx",
    )

    def test_sheets_in_order_with_the_hidden_flag(self):
        assert [(s.name, s.hidden) for s in self.BOOK.sheets] == [
            ("Kunden", False),
            ("Lookups", True),
        ]

    def test_a_formula_reads_empty_without_a_cached_value(self):
        book = parse(xlsx(("S", [["a", "b", "c"], [1, "=A2*2", "x"]])), "f.xlsx")
        table = book.sheets[0].table()
        # openpyxl writes formulas without a cached value, as a script would.
        assert table.frame["b"].isna().all()
        assert table.kinds["b"] == "empty"

    def test_sheets_past_the_cap_are_named_as_unread(self, monkeypatch):
        monkeypatch.setattr(reading, "_MAX_SHEETS", 2)
        book = parse(
            xlsx(("A", [["x"], [1]]), ("B", [["x"], [1]]), ("C", [["x"], [1]])),
            "w.xlsx",
        )
        assert [s.name for s in book.sheets] == ["A", "B"]
        assert (book.more_sheets, book.unread_sheets) == (1, ["C"])

    def test_a_calamine_failure_is_unreadable(self, monkeypatch):
        import python_calamine

        def broken(stream):
            raise python_calamine.CalamineError("boom")

        monkeypatch.setattr(python_calamine.CalamineWorkbook, "from_filelike", broken)
        with pytest.raises(SpreadsheetError, match="could not be read"):
            parse(xlsx(("A", [["x"]])), "w.xlsx")

    def test_reads_wait_for_a_free_slot_and_say_so_when_none_frees(self, monkeypatch):
        import threading

        monkeypatch.setattr(reading, "_parses", threading.BoundedSemaphore(1))
        monkeypatch.setattr(reading, "_PARSE_WAIT_SECONDS", 0.05)
        reading._parses.acquire()
        try:
            with pytest.raises(SpreadsheetError, match="too many spreadsheets"):
                parse(xlsx(("A", [["x"]])), "w.xlsx")
        finally:
            reading._parses.release()
        assert [s.name for s in parse(xlsx(("A", [["x"]])), "w.xlsx").sheets] == ["A"]


class TestTables:
    ORDERS = [
        ["Q3 orders"],
        [],
        ["Order ID", "Region", "Amount", "Kostenstelle", "Order date", "Paid"],
    ] + [
        [
            1000 + i,
            ["North", "South", "East", "North"][i % 4],
            [120.5, 80, 310.25, 45][i % 4],
            f"KST-44{10 + i % 2}",
            datetime.date(2026, 7, 1 + i),
            i % 3 == 0,
        ]
        for i in range(8)
    ]

    def sheet(self):
        return parse(xlsx(("Orders", self.ORDERS)), "w.xlsx").sheets[0]

    def test_types_are_inferred_per_column(self):
        assert self.sheet().table().kinds == {
            "Order ID": "integer",
            "Region": "text",
            "Amount": "number",
            "Kostenstelle": "text",
            "Order date": "date",
            "Paid": "boolean",
        }

    def test_a_number_column_with_no_text_is_not_an_integer_column(self):
        """An empty text slice once turned the float column to object dtype, and
        120.5 read as an integer column."""
        table = parse(xlsx(("S", [["a"], [1.5], [2.25]])), "w.xlsx").sheets[0].table()
        assert table.kinds["a"] == "number"

    def test_the_rows_above_a_header_are_kept(self):
        table = self.sheet().table()
        assert (table.header_row, table.above) == (3, [["Q3 orders"], []])

    def test_header_row_overrides_the_detected_one(self):
        table = self.sheet().table(header_row=4)
        assert list(table.frame.columns)[:2] == ["1000", "North"]
        assert not table.detected

    def test_only_one_chosen_header_row_is_kept_at_a_time(self):
        sheet = self.sheet()
        fourth = sheet.table(header_row=4)
        sheet.table(header_row=5)
        assert sheet.table(header_row=4) is not fourth
        assert sheet.table() is sheet.table()

    @pytest.mark.parametrize("row", [-1, 99])
    def test_a_header_row_off_the_sheet_says_the_range(self, row):
        sheet = parse(xlsx(("Kunden", KUNDEN)), "w.xlsx").sheets[0]
        with pytest.raises(QueryError) as raised:
            sheet.table(header_row=row)
        assert str(raised.value) == (
            f'header_row must be 0 (detected) or 1 to 3 for sheet "Kunden"; got {row}.'
        )

    def test_empty_and_repeated_header_cells_get_names(self):
        book = parse(
            xlsx(("S", [["Amount", None, "Amount", "Amount"], [1, 2, 3, 4]])), "w.xlsx"
        )
        assert list(book.sheets[0].table().frame.columns) == [
            "Amount",
            "column_2",
            "Amount (2)",
            "Amount (3)",
        ]

    def test_a_row_whose_only_cell_is_off_type_is_still_a_row(self):
        rows = [["n"]] + [[i] for i in range(10)] + [["Total"]]
        table = parse(xlsx(("S", rows)), "w.xlsx").sheets[0].table()
        assert (len(table.frame), table.kinds["n"], table.skipped["n"]) == (
            11,
            "integer",
            1,
        )

    def test_a_date_past_pandas_nanosecond_range_is_still_a_date(self):
        rows = [["valid until"], [datetime.date(2026, 1, 1)]]
        rows += [[datetime.date(9999, 12, 31)] for _ in range(3)]
        table = parse(xlsx(("S", rows)), "w.xlsx").sheets[0].table()
        assert (table.kinds["valid until"], table.skipped["valid until"]) == ("date", 0)
        assert str(table.frame["valid until"].max().date()) == "9999-12-31"


class TestText:
    def test_utf8_with_a_byte_order_mark(self):
        book = parse("﻿Name,City\nJosé,Köln\n".encode(), "a.csv")
        assert book.sheets[0].grid == [["Name", "City"], ["José", "Köln"]]

    def test_cp1252_is_detected(self):
        text = "Lieferant;Größe;Betrag\nMüller GmbH;groß;12,50 €\nBäckerei Öz;klein;3,00 €\n"
        book = parse(text.encode("cp1252"), "a.csv")
        assert book.sheets[0].grid[1] == ["Müller GmbH", "groß", "12,50 €"]

    def test_bytes_no_detector_reads_fall_back_to_cp1252(self, monkeypatch):
        class NoMatch:
            def best(self):
                return None

        monkeypatch.setattr(
            reading.charset_normalizer, "from_bytes", lambda data: NoMatch()
        )
        book = parse(b"a,b\n\x80,\xe9\n", "a.csv")
        assert book.sheets[0].grid[1] == ["€", "é"]

    @pytest.mark.parametrize("delimiter", [",", ";", "\t", "|"])
    def test_the_delimiter_is_sniffed(self, delimiter):
        lines = ["Report", "", delimiter.join(["a", "b", "c"]), delimiter.join("123")]
        book = parse("\n".join(lines).encode(), "a.csv")
        assert book.sheets[0].grid[2:] == [["a", "b", "c"], ["1", "2", "3"]]

    def test_european_amounts_do_not_make_a_semicolon_file_comma_separated(self):
        text = "Konto 4410, März\nTag;Text;Betrag\n01.03.;Porto;1.214,74\n02.03.;Miete;99,10\n"
        book = parse(text.encode("cp1252"), "a.csv")
        assert book.sheets[0].grid[1:] == [
            ["Tag", "Text", "Betrag"],
            ["01.03.", "Porto", "1.214,74"],
            ["02.03.", "Miete", "99,10"],
        ]

    def test_quoted_delimiters_do_not_split_a_field(self):
        book = parse(b'name,notes\nAcme,"late, again"\nBrix,"ok, fine"\n', "a.csv")
        assert book.sheets[0].grid[1] == ["Acme", "late, again"]

    def test_a_tsv_splits_on_tabs(self):
        book = parse(b"a,x\tb\n1,2\t3\n", "a.tsv")
        assert (book.kind, book.sheets[0].grid) == ("tsv", [["a,x", "b"], ["1,2", "3"]])

    def test_semicolon_files_read_european_numbers(self):
        table = parse(b"Betrag;Menge\n1.234,56;2\n99,10;1\n", "a.csv").sheets[0].table()
        assert (table.kinds["Betrag"], table.frame["Betrag"].tolist()) == (
            "number",
            [1234.56, 99.1],
        )

    def test_comma_files_read_thousands_separators(self):
        book = parse(b'Amount\n"1,234.50"\n12\n', "a.csv")
        assert book.sheets[0].table().frame["Amount"].tolist() == [1234.5, 12.0]

    def test_text_dates_become_dates_when_nine_in_ten_parse(self):
        rows = "\n".join(f"0{d}.03.2026;x" for d in range(1, 10))
        table = (
            parse(f"Datum;T\n{rows}\nkein Datum;y\n".encode(), "a.csv")
            .sheets[0]
            .table()
        )
        assert (table.kinds["Datum"], table.skipped["Datum"]) == ("date", 1)
        assert str(table.frame["Datum"].iloc[0].date()) == "2026-03-01"

    def test_day_first_slash_dates_in_a_comma_file_read_day_first(self):
        """Comma files once always read slash dates month-first, so 25/12 failed."""
        rows = "\n".join(f"{d}/12/2026,x" for d in range(13, 31))
        table = parse(f"when,t\n{rows}\n".encode(), "a.csv").sheets[0].table()
        assert table.kinds["when"] == "date"
        assert str(table.frame["when"].iloc[0].date()) == "2026-12-13"

    def test_ambiguous_slash_dates_in_a_comma_file_read_month_first(self):
        table = parse(b"when\n01/02/2026\n03/04/2026\n", "a.csv").sheets[0].table()
        assert str(table.frame["when"].iloc[0].date()) == "2026-01-02"

    def test_mostly_text_with_some_numbers_is_mixed(self):
        assert (
            parse(b"code\nA1\n2\nB7\n3\n", "a.csv").sheets[0].table().kinds["code"]
            == "mixed"
        )

    def test_columns_past_the_cap_are_cut_and_counted(self, monkeypatch):
        monkeypatch.setattr(reading, "_MAX_COLUMNS", 3)
        sheet = parse(b"a,b,c,d,e\n1,2,3,4,5\n", "a.csv").sheets[0]
        assert (sheet.grid[0], sheet.more_columns) == (["a", "b", "c"], 2)

    def test_a_field_past_the_csv_limit_is_unreadable(self):
        assert reason(b'a\n"' + b"x" * 200_000 + b'"\n', "a.csv") == "unreadable"

    def test_a_csv_past_the_cell_budget_is_refused(self, monkeypatch):
        monkeypatch.setattr(reading, "_MAX_CELLS", 100)
        with pytest.raises(SpreadsheetError, match="more than 100 cells"):
            parse(b"a,b\n" * 60, "a.csv")


class TestHeaderDetection:
    def test_title_rows_above_the_header_are_skipped(self):
        text = "Umsatzbericht März 2026\nErstellt am 02.04.2026\n\nDatum;Text;;Betrag\n03.03.2026;Miete;;1.234,56\n"
        assert parse(text.encode(), "a.csv").sheets[0].table().header_row == 4

    def test_a_row_of_numbers_is_never_the_header(self):
        assert parse(b"2026;2027\n1;2\n", "a.csv").sheets[0].table().header_row == 1

    def test_only_the_first_twenty_rows_are_scanned(self):
        rows = ["1,2"] * 25 + ["Name,City"]
        assert (
            parse("\n".join(rows).encode(), "a.csv").sheets[0].table().header_row == 1
        )


def test_exact_answers_over_two_hundred_thousand_rows():
    """Counts and totals come from every row, not a sample."""
    lines = ["region,amount"] + [
        f"{['North', 'South'][i % 2]},{i % 100}" for i in range(200_000)
    ]
    table = parse("\n".join(lines).encode(), "big.csv").sheets[0].table()
    assert len(table.frame) == 200_000
    assert int(table.frame["amount"].sum()) == sum(i % 100 for i in range(200_000))
    assert int((table.frame["region"] == "North").sum()) == 100_000


class TestScanAgreesWithCalamine:
    """The pre-scan must never measure a sheet smaller than calamine fills it."""

    @pytest.mark.parametrize(
        "cells",
        [
            [(1, 1)],
            [(3, 2), (1, 5)],
            [(1, 1), (500, 3)],
            [(2, 30), (40, 1)],
            [(1, 1), (7, 7), (7, 8)],
        ],
        ids=["a1", "scattered", "tall", "wide", "corner"],
    )
    def test_its_extent_is_calamines(self, cells):
        import python_calamine

        book = openpyxl.Workbook()
        sheet = book.active
        sheet.title = "S"
        for row, column in cells:
            sheet.cell(row=row, column=column, value=f"v{row}-{column}")
        out = io.BytesIO()
        book.save(out)
        data = out.getvalue()

        measured = reading._measure_archive(data)["S"]
        end = (
            python_calamine.CalamineWorkbook.from_filelike(io.BytesIO(data))
            .get_sheet_by_name("S")
            .end
        )
        assert measured == (end[0] + 1, end[1] + 1)

    def test_cells_without_positions_land_where_calamine_puts_them(self):
        import python_calamine

        rows = (
            b'<row r="2"><c t="inlineStr"><is><t>a</t></is></c><c/>'
            b'<c t="n"><v>1</v></c></row><row><c/><c t="n"><v>2</v></c></row>'
        )
        data = with_cells(rows)
        end = (
            python_calamine.CalamineWorkbook.from_filelike(io.BytesIO(data))
            .get_sheet_by_name("Data")
            .end
        )
        assert reading._measure_archive(data)["Data"] == (end[0] + 1, end[1] + 1)


_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELS_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
_DOC_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def package(
    folder: str, sheets: dict[str, bytes], decoy: bool = False, **extra
) -> bytes:
    """A workbook in ``folder`` whose ``_rels/.rels`` points there, holding
    ``sheets`` by name; ``decoy`` adds a one-cell workbook at the usual ``xl/``.
    ``extra`` adds members as written."""

    def book(at: str, named: dict[str, bytes]) -> dict[str, bytes]:
        entries = "".join(
            f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(named, 1)
        )
        rels = "".join(
            f'<Relationship Id="rId{i}" Type="{_DOC_NS}/worksheet" '
            f'Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(named) + 1)
        )
        return {
            f"{at}workbook.xml": (
                f'<workbook xmlns="{_SHEET_NS}" xmlns:r="{_DOC_NS}"><sheets>{entries}'
                "</sheets></workbook>"
            ).encode(),
            f"{at}_rels/workbook.xml.rels": (
                f'<Relationships xmlns="{_RELS_NS}">{rels}</Relationships>'
            ).encode(),
            **{
                f"{at}worksheets/sheet{i}.xml": (
                    f'<worksheet xmlns="{_SHEET_NS}"><sheetData>'.encode()
                    + rows
                    + b"</sheetData></worksheet>"
                )
                for i, rows in enumerate(named.values(), 1)
            },
        }

    members = {
        "_rels/.rels": (
            f'<Relationships xmlns="{_RELS_NS}"><Relationship Id="rId1" '
            f'Type="{_DOC_NS}/officeDocument" Target="{folder}workbook.xml"/>'
            "</Relationships>"
        ).encode(),
        **book(folder, sheets),
    }
    if decoy:
        members.update(book("xl/", {name: CELL_DECOY for name in sheets}))
    members.update(extra)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return out.getvalue()


def inline(reference: str, text: str) -> bytes:
    return (
        f'<row r="{reference.lstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}">'
        f'<c r="{reference}" t="inlineStr"><is><t>{text}</t></is></c></row>'
    ).encode()


CELL_DECOY = inline("A1", "decoy")


class TestScanFollowsCalaminesParts:
    """The scan must open the members calamine opens, or it measures a decoy."""

    def test_the_workbook_the_package_points_to_is_the_one_read(self):
        book = parse(
            package("other/", {"S": inline("A1", "real")}, decoy=True), "w.xlsx"
        )
        assert book.sheets[0].grid == [["real"]]

    def test_a_far_sheet_behind_a_redirected_package_is_refused_unopened(
        self, calamine_must_not_open
    ):
        data = package("other/", {"S": inline("XFD1048576", "far")}, decoy=True)
        assert reason(data) == "too_far"

    def test_a_backslash_target_reads_the_root_workbook_as_calamine_does(self):
        """calamine takes the workbook folder up to the last "/" only, so a target
        of ``xl\\workbook.xml`` names the root ``workbook.xml``, not ``xl/``'s."""
        data = backslashed(
            root=inline("A1", "root"), under_xl=inline("XFD1048576", "x")
        )
        assert parse(data, "w.xlsx").sheets[0].grid == [["root"]]

    def test_a_backslash_target_measures_the_root_workbook_calamine_would_read(
        self, calamine_must_not_open
    ):
        data = backslashed(
            root=inline("XFD1048576", "x"), under_xl=inline("A1", "small")
        )
        assert reason(data) == "too_far"

    def test_members_whose_names_differ_only_in_case_are_refused(self):
        data = package(
            "xl/",
            {"S": CELL_DECOY},
            **{"xl/worksheets/SHEET1.xml": b"<worksheet/>"},
        )
        with pytest.raises(SpreadsheetError, match="could not be read"):
            reading._measure_archive(data)

    def test_two_sheets_with_one_name_are_refused(self):
        data = package("xl/", {"S": CELL_DECOY, "T": CELL_DECOY})
        data = rewrite_member(
            data, "xl/workbook.xml", lambda xml: xml.replace(b'"T"', b'"S"')
        )
        with pytest.raises(SpreadsheetError, match="could not be read"):
            reading._measure_archive(data)

    def test_a_target_with_xml_escapes_is_refused(self):
        data = package("xl/", {"S": CELL_DECOY})
        data = rewrite_member(
            data,
            "xl/_rels/workbook.xml.rels",
            lambda xml: xml.replace(b"worksheets/", b"work&amp;sheets/"),
        )
        with pytest.raises(SpreadsheetError, match="could not be read"):
            reading._measure_archive(data)


def rewrite_member(data: bytes, member: str, change) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    parts[member] = change(parts[member])
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return out.getvalue()


def test_a_csv_waits_for_a_parse_slot_like_an_xlsx(monkeypatch):
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(reading, "_parses", slots)
    monkeypatch.setattr(reading, "_PARSE_WAIT_SECONDS", 0.01)

    assert reason(b"a,b\n1,2\n", "a.csv") == "busy"


def test_a_header_row_asked_for_is_the_one_returned_while_another_is_cached(
    monkeypatch,
):
    """Another caller caching a different header row right after this one's table
    is cached must not swap the table this caller gets back."""
    sheet = parse(b"a,b\nc,d\n1,2\n3,4\n", "a.csv").sheets[0]
    held: dict[str, object] = {}

    def cache(target, table):
        held["table"] = table
        if table is not None and table.header_row == 1 and "raced" not in held:
            held["raced"] = True
            target.table(header_row=2)

    monkeypatch.setattr(
        reading.Sheet, "_chosen", property(lambda _: held.get("table"), cache)
    )

    assert sheet.table(header_row=1).header_row == 1


def merge(first: bytes, second: bytes) -> bytes:
    """Both archives' members in one; the second's win on a shared name."""
    parts: dict[str, bytes] = {}
    for data in (first, second):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            parts.update({name: archive.read(name) for name in archive.namelist()})
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in parts.items():
            archive.writestr(name, content)
    return out.getvalue()


def backslashed(root: bytes, under_xl: bytes) -> bytes:
    """A workbook at the root and one under ``xl/``, with ``_rels/.rels`` naming
    ``xl\\workbook.xml``."""
    rels = (
        f'<Relationships xmlns="{_RELS_NS}"><Relationship Id="rId1" '
        f'Type="{_DOC_NS}/officeDocument" Target="xl\\workbook.xml"/>'
        "</Relationships>"
    ).encode()
    return merge(
        package("xl/", {"S": under_xl}),
        package("", {"S": root}, **{"_rels/.rels": rels}),
    )
