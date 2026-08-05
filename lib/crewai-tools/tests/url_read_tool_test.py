from unittest.mock import patch

import pytest
import requests

from crewai_tools import URLReadTool
from crewai_tools.security.safe_requests import safe_get_bounded


TOOL_MODULE = "crewai_tools.tools.url_read_tool.url_read_tool"


class FakeResponse:
    """Minimal stand-in for a streamed requests.Response."""

    def __init__(
        self,
        body: bytes = b"",
        content_type: str = "text/plain",
        url: str = "https://example.com/file.txt",
        status_code: int = 200,
        chunk_size: int | None = None,
    ):
        self._body = body
        self._chunk_size = chunk_size
        self.headers = {"Content-Type": content_type} if content_type else {}
        self.url = url
        self.status_code = status_code
        self.history: list["FakeResponse"] = []
        self.closed = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def iter_content(self, chunk_size: int = 65536):
        size = self._chunk_size or chunk_size
        for index in range(0, len(self._body), size):
            yield self._body[index : index + size]

    def close(self) -> None:
        self.closed = True


def fetch_result(
    body: bytes, content_type: str = "text/plain", url: str = "https://example.com/f.txt"
):
    """Build the (body, content_type, final_url) tuple safe_get_bounded returns."""
    return body, content_type, url


def test_reads_plain_text():
    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(b"hello world")
        assert tool.run(url="https://example.com/f.txt") == "hello world"

    assert fetch.call_args.kwargs["max_bytes"] == 5 * 1024 * 1024
    assert fetch.call_args.kwargs["timeout"] == 30


def test_honors_declared_charset():
    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(
            "café".encode("latin-1"), "text/plain; charset=iso-8859-1"
        )
        assert tool.run(url="https://example.com/f.txt") == "café"


def test_encoding_override_wins_over_server_charset():
    tool = URLReadTool(encoding="latin-1")
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(
            "café".encode("latin-1"), "text/plain; charset=utf-8"
        )
        assert tool.run(url="https://example.com/f.txt") == "café"


def test_undecodable_bytes_fall_back_instead_of_failing():
    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(b"\xff\xfe bad bytes", "text/plain")
        result = tool.run(url="https://example.com/f.txt")

    assert "bad bytes" in result
    assert not result.startswith("Error:")


def test_line_window():
    tool = URLReadTool()
    body = b"one\ntwo\nthree\nfour\nfive\n"
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(body)
        result = tool.run(url="https://example.com/f.txt", start_line=2, line_count=2)

    assert result == "two\nthree\n"


def test_start_line_past_end_reports_error():
    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(b"one\ntwo\n")
        result = tool.run(url="https://example.com/f.txt", start_line=99)

    assert "exceeds the number of lines" in result


def test_json_is_returned_verbatim():
    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(b'{"a": 1}', "application/json")
        assert tool.run(url="https://example.com/data.json") == '{"a": 1}'


def test_structured_suffix_type_is_treated_as_text():
    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(b'{"a": 1}', "application/vnd.api+json")
        assert tool.run(url="https://example.com/data") == '{"a": 1}'


def test_html_is_stripped_to_visible_text():
    tool = URLReadTool()
    body = b"<html><head><style>p{color:red}</style></head><body><p>Hi</p><script>x=1</script></body></html>"
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(body, "text/html; charset=utf-8")
        result = tool.run(url="https://example.com/page")

    assert "Hi" in result
    assert "x=1" not in result
    assert "color:red" not in result


def test_binary_content_type_is_rejected():
    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(b"\x89PNG\r\n", "image/png")
        result = tool.run(url="https://example.com/logo.png")

    assert "Unsupported content type 'image/png'" in result


def test_octet_stream_falls_back_to_url_extension():
    tool = URLReadTool()
    assert (
        tool._resolve_kind("application/octet-stream", "https://example.com/a/b.pdf")
        == "pdf"
    )
    assert tool._resolve_kind("", "https://example.com/a/b.csv") == "text"
    assert tool._resolve_kind("", "https://example.com/a/b.bin") is None


def test_query_string_does_not_break_extension_fallback():
    tool = URLReadTool()
    assert (
        tool._resolve_kind("application/octet-stream", "https://example.com/b.pdf?v=2")
        == "pdf"
    )


def test_validation_failure_is_returned_as_error():
    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.side_effect = ValueError(
            "URL 'http://169.254.169.254/' resolves to private/reserved IP 169.254.169.254."
        )
        result = tool.run(url="http://169.254.169.254/")

    assert result.startswith("Error:")
    assert "private/reserved IP" in result


def test_request_failure_is_returned_as_error():
    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.side_effect = requests.ConnectionError("connection refused")
        result = tool.run(url="https://example.com/f.txt")

    assert result.startswith("Error: Failed to fetch")


def test_custom_headers_are_merged_over_defaults():
    tool = URLReadTool(headers={"Authorization": "Bearer x", "Accept": "text/plain"})
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(b"ok")
        tool.run(url="https://example.com/f.txt")

    headers = fetch.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer x"
    assert headers["Accept"] == "text/plain"
    assert "crewai-tools URLReadTool" in headers["User-Agent"]


def test_reads_a_real_pdf_end_to_end():
    pymupdf = pytest.importorskip("pymupdf")

    document = pymupdf.open()
    document.new_page().insert_text((72, 72), "Quarterly revenue was 42")
    pdf_bytes = document.tobytes()
    document.close()

    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(
            pdf_bytes, "application/pdf", "https://example.com/report.pdf"
        )
        result = tool.run(url="https://example.com/report.pdf")

    assert "Page 1:" in result
    assert "Quarterly revenue was 42" in result


def test_corrupt_pdf_reports_error_without_raising():
    pytest.importorskip("pymupdf")

    tool = URLReadTool()
    with patch(f"{TOOL_MODULE}.safe_get_bounded") as fetch:
        fetch.return_value = fetch_result(b"%PDF-1.4 not really a pdf", "application/pdf")
        result = tool.run(url="https://example.com/report.pdf")

    assert result.startswith("Error: Failed to read PDF content")


class TestSafeGetBounded:
    """Tests for the bounded-fetch helper itself."""

    def test_returns_body_content_type_and_final_url(self):
        response = FakeResponse(b"payload", "text/plain", "https://example.com/final")
        with patch(
            "crewai_tools.security.safe_requests.safe_get", return_value=response
        ):
            body, content_type, final_url = safe_get_bounded(
                "https://example.com/start", max_bytes=1024
            )

        assert body == b"payload"
        assert content_type == "text/plain"
        assert final_url == "https://example.com/final"
        assert response.closed

    def test_rejects_body_over_the_limit(self):
        response = FakeResponse(b"x" * 100, chunk_size=10)
        with patch(
            "crewai_tools.security.safe_requests.safe_get", return_value=response
        ):
            with pytest.raises(ValueError, match="exceeds the 25 byte limit"):
                safe_get_bounded("https://example.com/big", max_bytes=25)

        assert response.closed

    def test_stops_reading_once_the_limit_is_crossed(self):
        """The cap must abandon the stream, not buffer the whole body first."""
        chunks_yielded = 0

        class CountingResponse(FakeResponse):
            def iter_content(self, chunk_size: int = 65536):
                nonlocal chunks_yielded
                for _ in range(1000):
                    chunks_yielded += 1
                    yield b"x" * 10

        response = CountingResponse()
        with patch(
            "crewai_tools.security.safe_requests.safe_get", return_value=response
        ):
            with pytest.raises(ValueError):
                safe_get_bounded("https://example.com/huge", max_bytes=25)

        assert chunks_yielded == 3

    def test_error_status_raises(self):
        response = FakeResponse(b"nope", status_code=404)
        with patch(
            "crewai_tools.security.safe_requests.safe_get", return_value=response
        ):
            with pytest.raises(requests.HTTPError):
                safe_get_bounded("https://example.com/missing", max_bytes=1024)

        assert response.closed

    def test_closes_redirect_hops(self):
        hop = FakeResponse(b"", status_code=302)
        response = FakeResponse(b"done")
        response.history = [hop]
        with patch(
            "crewai_tools.security.safe_requests.safe_get", return_value=response
        ):
            safe_get_bounded("https://example.com/start", max_bytes=1024)

        assert hop.closed
        assert response.closed

    def test_requests_are_streamed(self):
        response = FakeResponse(b"ok")
        with patch(
            "crewai_tools.security.safe_requests.safe_get", return_value=response
        ) as safe_get:
            safe_get_bounded("https://example.com/f", max_bytes=1024)

        assert safe_get.call_args.kwargs["stream"] is True
