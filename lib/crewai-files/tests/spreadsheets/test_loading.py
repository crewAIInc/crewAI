"""Recognising a spreadsheet whatever file type it arrived as, and reading it once
within its byte limit, from a path, bytes or a URL."""

from __future__ import annotations

import io
import subprocess
import sys
import tracemalloc
import zipfile

from crewai_files import AudioFile, File, FileBytes, FilePath, FileUrl, TextFile
import crewai_files.spreadsheets as spreadsheets
import crewai_files.spreadsheets.reading as reading
import openpyxl
import pytest


def xlsx_bytes() -> bytes:
    book = openpyxl.Workbook()
    book.active.title = "Orders"
    book.active.append(["Region", "Amount"])
    book.active.append(["North", 10])
    out = io.BytesIO()
    book.save(out)
    return out.getvalue()


CSV = b"region,amount\nNorth,10\nSouth,20\n"


class TestIsSpreadsheet:
    @pytest.mark.parametrize(
        "file",
        [
            TextFile(source=FileBytes(data=CSV, filename="sales.csv")),
            File(source=FileBytes(data=CSV, filename="sales.TSV")),
            # A JSON crew's {"source": "data.csv"} validates as an AudioFile.
            AudioFile(source=FileBytes(data=CSV, filename="data.csv")),
            File(source=FileBytes(data=xlsx_bytes(), filename="book.xlsx")),
            File(source=FileBytes(data=b"\xd0\xcf\x11\xe0", filename="old.xls")),
            File(source=FileBytes(data=b"PK", filename="budget.ods")),
            File(source=FileUrl(url="https://bucket.example/u/42/sales.xlsx?sig=abc")),
        ],
        ids=["text-csv", "file-tsv", "audio-csv", "xlsx", "xls", "ods", "url"],
    )
    def test_every_way_a_spreadsheet_arrives(self, file):
        assert spreadsheets.is_spreadsheet(file)

    def test_an_unnamed_workbook_is_found_inside_its_archive(self):
        assert spreadsheets.is_spreadsheet(File(source=FileBytes(data=xlsx_bytes())))

    @pytest.mark.parametrize(
        "file",
        [
            TextFile(source=FileBytes(data=b"hello", filename="notes.txt")),
            File(source=FileBytes(data=b"%PDF-1.7", filename="report.pdf")),
            File(source=FileUrl(url="https://bucket.example/u/42/download?id=1")),
        ],
        ids=["text", "pdf", "url-without-a-name"],
    )
    def test_other_files_are_not_spreadsheets(self, file):
        assert not spreadsheets.is_spreadsheet(file)

    def test_a_url_is_never_downloaded_to_decide(self, monkeypatch):
        def refuse(*args, **kwargs):
            raise AssertionError("must not download")

        monkeypatch.setattr("httpx.get", refuse)
        monkeypatch.setattr("httpx.stream", refuse)
        spreadsheets.is_spreadsheet(
            File(source=FileUrl(url="https://bucket.example/x"))
        )

    def test_recognising_an_archive_reads_only_the_start_of_its_mimetype(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("mimetype", b"a" * 200_000_000)
        bomb = File(source=FileBytes(data=out.getvalue()))

        tracemalloc.start()
        try:
            assert not spreadsheets.is_spreadsheet(bomb)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < 10_000_000

    def test_an_archive_of_too_many_members_is_not_opened_further(self):
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as archive:
            for i in range(spreadsheets._MAX_MEMBERS + 1):
                archive.writestr(f"m{i}", b"")
            archive.writestr("xl/workbook.xml", b"")
        assert not spreadsheets.is_spreadsheet(
            File(source=FileBytes(data=out.getvalue()))
        )

    def test_a_damaged_archive_is_not_a_spreadsheet(self):
        damaged = File(source=FileBytes(data=b"PK\x03\x04" + b"\x00" * 64))
        assert not spreadsheets.is_spreadsheet(damaged)

    def test_importing_it_needs_no_pandas(self):
        """crewAI routes files with is_spreadsheet whether or not the extra is
        installed."""
        probe = (
            "import sys, crewai_files.spreadsheets as s; "
            "assert 'pandas' not in sys.modules, 'pandas imported'; "
            "assert 'python_calamine' not in sys.modules"
        )
        result = subprocess.run(  # noqa: S603 - this interpreter on a fixed probe
            [sys.executable, "-c", probe], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr


class TestLoad:
    def test_a_csv_from_bytes(self):
        book = spreadsheets.load_spreadsheet(
            TextFile(source=FileBytes(data=CSV, filename="s.csv"))
        )
        assert (book.kind, len(book.sheets[0].table().frame)) == ("csv", 2)

    def test_the_same_content_is_read_once(self, monkeypatch):
        calls = []
        real = reading.parse

        def counting(data, name, *, max_bytes):
            calls.append(name)
            return real(data, name, max_bytes=max_bytes)

        monkeypatch.setattr(reading, "parse", counting)
        data = CSV + b"East,99\n"
        first = spreadsheets.load_spreadsheet(
            File(source=FileBytes(data=data, filename="a.csv"))
        )
        again = spreadsheets.load_spreadsheet(
            File(source=FileBytes(data=data, filename="b.csv"))
        )
        assert first is again
        assert calls == ["a.csv"]

    def test_unnamed_bytes_typed_as_csv_read_as_csv(self, monkeypatch):
        file = File(source=FileBytes(data=CSV))
        monkeypatch.setattr(
            FileBytes, "content_type", property(lambda self: "text/csv")
        )
        assert spreadsheets.load_spreadsheet(file).kind == "csv"

    def test_a_local_file_over_the_limit_is_refused_unread(self, tmp_path, monkeypatch):
        path = tmp_path / "big.csv"
        path.write_bytes(b"a,b\n" * 1000)

        def refuse(self):
            raise AssertionError("must not read")

        monkeypatch.setattr(FilePath, "read", refuse)
        with pytest.raises(reading.SpreadsheetError, match=r"over the 0 MB limit"):
            spreadsheets.load_spreadsheet(File(source=path), max_bytes=100)


class _Response:
    def __init__(self, chunks: list[bytes], length: int | None = None, status=200):
        self.chunks = chunks
        self.headers = {} if length is None else {"content-length": str(length)}
        self.status = status
        self.read = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        import httpx

        if self.status >= 400:
            raise httpx.HTTPStatusError(
                "nope", request=httpx.Request("GET", "https://x"), response=None
            )

    def iter_bytes(self):
        for chunk in self.chunks:
            self.read += len(chunk)
            yield chunk


class TestDownload:
    URL = "https://bucket.example/u/42/sales.csv"

    def serve(self, monkeypatch, response):
        monkeypatch.setattr("httpx.stream", lambda *a, **k: response)
        return response

    def test_a_url_is_streamed_and_read(self, monkeypatch):
        self.serve(monkeypatch, _Response([CSV[:10], CSV[10:]]))
        assert (
            spreadsheets.load_spreadsheet(File(source=FileUrl(url=self.URL))).kind
            == "csv"
        )

    def test_a_declared_length_over_the_limit_is_refused_before_reading(
        self, monkeypatch
    ):
        response = self.serve(monkeypatch, _Response([b"x" * 10], length=10**9))
        with pytest.raises(reading.SpreadsheetError, match="over the 1 MB limit"):
            spreadsheets.load_spreadsheet(
                File(source=FileUrl(url=self.URL)), max_bytes=1_048_576
            )
        assert response.read == 0

    def test_a_body_past_the_limit_stops_the_download(self, monkeypatch):
        chunks = [b"a,b\n" * 100] * 1000
        response = self.serve(monkeypatch, _Response(chunks))
        with pytest.raises(reading.SpreadsheetError, match="over the 0 MB limit"):
            spreadsheets.load_spreadsheet(
                File(source=FileUrl(url=self.URL)), max_bytes=1_000
            )
        assert response.read < 2_000

    def test_a_failed_download_says_so(self, monkeypatch):
        self.serve(monkeypatch, _Response([], status=404))
        with pytest.raises(
            reading.SpreadsheetError, match="could not be downloaded"
        ) as raised:
            spreadsheets.load_spreadsheet(File(source=FileUrl(url=self.URL)))
        assert raised.value.reason == "download"


def test_the_workbook_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(spreadsheets, "_workbooks", type(spreadsheets._workbooks)())
    for i in range(spreadsheets._CACHED_WORKBOOKS + 3):
        data = f"a\n{i}\n".encode()
        spreadsheets.load_spreadsheet(
            File(source=FileBytes(data=data, filename="a.csv"))
        )
    assert len(spreadsheets._workbooks) == spreadsheets._CACHED_WORKBOOKS


def test_the_workbook_cache_holds_cells_up_to_its_budget(monkeypatch):
    monkeypatch.setattr(spreadsheets, "_workbooks", type(spreadsheets._workbooks)())
    monkeypatch.setattr(spreadsheets, "_CACHED_CELLS", 5)
    for i in range(2):
        data = f"a,b\n{i},{i}\n".encode()  # 4 cells
        spreadsheets.load_spreadsheet(
            File(source=FileBytes(data=data, filename="a.csv"))
        )
    assert len(spreadsheets._workbooks) == 1


def test_a_workbook_over_the_cell_budget_is_still_kept_while_newest(monkeypatch):
    monkeypatch.setattr(spreadsheets, "_workbooks", type(spreadsheets._workbooks)())
    monkeypatch.setattr(spreadsheets, "_CACHED_CELLS", 1)
    data = b"a,b\n1,2\n"
    first = spreadsheets.load_spreadsheet(
        File(source=FileBytes(data=data, filename="a.csv"))
    )
    again = spreadsheets.load_spreadsheet(
        File(source=FileBytes(data=data, filename="b.csv"))
    )
    assert first is again
