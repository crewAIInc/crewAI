"""Tests for SkimReaderTool.

These tests are fully mocked: no network calls and no real x402 payments are
made. The payment-aware session and the URL validator are stubbed so the tests
are deterministic and offline.
"""

import pytest

from crewai_tools.tools.skim_reader_tool import skim_reader_tool as skim_module
from crewai_tools.tools.skim_reader_tool.skim_reader_tool import SkimReaderTool


class _FakeResponse:
    def __init__(self, *, status_code=200, json_data=None, text="", reason="OK"):
        self.status_code = status_code
        self.ok = status_code < 400
        self._json_data = json_data
        self.text = text
        self.reason = reason

    def json(self):
        if self._json_data is None:
            raise ValueError("no json")
        return self._json_data


class _FakeSession:
    def __init__(self, response):
        self._response = response
        self.calls = []

    def post(self, endpoint, json=None, timeout=None):
        self.calls.append({"endpoint": endpoint, "json": json, "timeout": timeout})
        return self._response


@pytest.fixture(autouse=True)
def _skip_url_validation(monkeypatch):
    # validate_url performs DNS resolution / SSRF checks; bypass it in unit tests.
    monkeypatch.setattr(skim_module, "validate_url", lambda url: url)


def _make_tool(response, **kwargs):
    tool = SkimReaderTool(**kwargs)
    tool._session = _FakeSession(response)
    return tool


def test_defaults_and_schema():
    tool = SkimReaderTool()
    assert tool.name == "Skim web reader"
    assert tool.base_url == "https://skim402.com"
    assert tool.max_price_usd == 0.01
    assert tool.include_metadata is True
    assert tool.timeout == 60.0
    assert tool.args_schema is not None
    assert "url" in tool.args_schema.model_fields


def test_run_returns_markdown_with_frontmatter():
    response = _FakeResponse(
        json_data={
            "markdown": "# Hello\n\nBody.",
            "metadata": {"title": "Hello", "language": "en", "empty": ""},
        }
    )
    tool = _make_tool(response)

    result = tool._run(url="https://example.com")

    assert result.startswith("---\n")
    assert "title: Hello" in result
    assert "language: en" in result
    assert "empty:" not in result  # empty values are dropped
    assert result.endswith("# Hello\n\nBody.")
    # Posts to the read endpoint in basic mode.
    call = tool._session.calls[0]
    assert call["endpoint"] == "https://skim402.com/api/v1/read"
    assert call["json"] == {"url": "https://example.com", "mode": "basic"}


def test_run_without_metadata():
    response = _FakeResponse(
        json_data={"markdown": "# Hello", "metadata": {"title": "Hello"}}
    )
    tool = _make_tool(response, include_metadata=False)

    result = tool._run(url="https://example.com")

    assert result == "# Hello"


def test_run_falls_back_to_text():
    response = _FakeResponse(json_data={"text": "plain text", "metadata": {}})
    tool = _make_tool(response)

    assert tool._run(url="https://example.com") == "plain text"


def test_run_raises_on_error_status():
    response = _FakeResponse(status_code=502, text="bad gateway", reason="Bad Gateway")
    tool = _make_tool(response)

    with pytest.raises(RuntimeError, match="502"):
        tool._run(url="https://example.com")


def test_get_session_requires_a_key(monkeypatch):
    monkeypatch.delenv("SKIM_WALLET_PRIVATE_KEY", raising=False)
    tool = SkimReaderTool()

    with pytest.raises(ValueError, match="SKIM_WALLET_PRIVATE_KEY"):
        tool._get_session()


def test_get_session_rejects_malformed_key(monkeypatch):
    monkeypatch.setenv("SKIM_WALLET_PRIVATE_KEY", "not-a-valid-hex-key")
    tool = SkimReaderTool()

    with pytest.raises(ValueError, match="64-character hex"):
        tool._get_session()
