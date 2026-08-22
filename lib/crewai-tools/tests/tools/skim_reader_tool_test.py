"""Unit tests for SkimReaderTool.

These tests inject a fake payment-aware session (via the cached ``_session``
attribute), so they never touch the network or sign a real payment.
"""

import pytest

from crewai_tools import SkimReaderTool
from crewai_tools.tools.skim_reader_tool.skim_reader_tool import _yaml_scalar

VALID_KEY = "0x" + "ab" * 32


class _FakeResp:
    def __init__(self, status=200, payload=None, text="", reason="OK"):
        self.status_code = status
        self._payload = payload or {}
        self.text = text
        self.reason = reason
        self.ok = 200 <= status < 300

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        return self._resp


def test_run_returns_markdown_with_frontmatter():
    tool = SkimReaderTool(private_key=VALID_KEY)
    fake = _FakeSession(
        _FakeResp(
            payload={
                "markdown": "# Title\n\nBody text.",
                "metadata": {
                    "title": "Title",
                    "byline": "Jane Doe",
                    "lang": "en",
                    "excerpt": "",  # empty values are dropped
                    "siteName": None,  # None values are dropped
                },
            }
        )
    )
    tool._session = fake

    out = tool._run(url="https://example.com/a")

    assert out.startswith("---\n")
    assert "title: Title" in out
    assert "byline: Jane Doe" in out
    assert "lang: en" in out
    assert "excerpt:" not in out
    assert "siteName:" not in out
    assert out.rstrip().endswith("Body text.")

    call = fake.calls[0]
    assert call["url"].endswith("/api/v1/read")
    assert call["json"] == {"url": "https://example.com/a", "mode": "basic"}


def test_include_metadata_false_returns_plain_markdown():
    tool = SkimReaderTool(private_key=VALID_KEY, include_metadata=False)
    tool._session = _FakeSession(
        _FakeResp(payload={"markdown": "# Title", "metadata": {"title": "Title"}})
    )

    assert tool._run(url="https://example.com/a") == "# Title"


def test_falls_back_to_text_when_no_markdown():
    tool = SkimReaderTool(private_key=VALID_KEY, include_metadata=False)
    tool._session = _FakeSession(_FakeResp(payload={"text": "plain text"}))

    assert tool._run(url="https://example.com/a") == "plain text"


def test_custom_base_url_is_used():
    tool = SkimReaderTool(private_key=VALID_KEY, base_url="https://example.test/")
    fake = _FakeSession(_FakeResp(payload={"markdown": "x"}))
    tool._session = fake

    tool._run(url="https://example.com/a")

    assert fake.calls[0]["url"] == "https://example.test/api/v1/read"


def test_http_error_raises_runtime_error():
    tool = SkimReaderTool(private_key=VALID_KEY)
    tool._session = _FakeSession(
        _FakeResp(status=502, text="upstream boom", reason="Bad Gateway")
    )

    with pytest.raises(RuntimeError) as exc:
        tool._run(url="https://example.com/a")

    assert "502" in str(exc.value)


def test_non_json_response_raises_runtime_error():
    tool = SkimReaderTool(private_key=VALID_KEY)

    class _BadJsonResp(_FakeResp):
        def json(self):
            raise ValueError("Expecting value")

    tool._session = _FakeSession(_BadJsonResp(text="<html>oops</html>"))

    with pytest.raises(RuntimeError):
        tool._run(url="https://example.com/a")


def test_yaml_scalar_quotes_ambiguous_values():
    assert _yaml_scalar("plain title") == "plain title"
    assert _yaml_scalar("key: value").startswith('"')
    assert _yaml_scalar("") == '""'
    assert _yaml_scalar("multi\nline\ntext") == "multi line text"


def test_tool_metadata_is_set():
    tool = SkimReaderTool(private_key=VALID_KEY)
    assert tool.name == "Skim web reader"
    assert "x402" in tool.description
    assert tool.args_schema is not None
