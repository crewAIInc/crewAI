"""Tests for SkimReaderTool.

These tests are fully mocked: no network calls, card charges, or x402 payments
are made. HTTP sessions and URL validation are stubbed for deterministic tests.
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
    """Record fake GET and POST requests without network access."""

    def __init__(self, response):
        self._response = response
        self.calls = []
        self.headers = {}

    def get(self, endpoint, params=None, timeout=None):
        self.calls.append(
            {"method": "GET", "endpoint": endpoint, "params": params, "timeout": timeout}
        )
        return self._response

    def post(self, endpoint, json=None, timeout=None):
        self.calls.append(
            {"method": "POST", "endpoint": endpoint, "json": json, "timeout": timeout}
        )
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
    assert tool.api_key is None
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
    assert call["method"] == "POST"
    assert call["endpoint"] == "https://skim402.com/api/v1/read"
    assert call["json"] == {"url": "https://example.com", "mode": "basic"}
    assert call["timeout"] == 60.0


def test_get_session_prefers_card_api_key(monkeypatch):
    """A card key builds a Bearer session without importing wallet packages."""
    fake = _FakeSession(_FakeResponse(json_data={"markdown": "ok"}))
    monkeypatch.setattr(skim_module.requests, "Session", lambda: fake)
    monkeypatch.setattr(
        skim_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError(name)),
    )

    tool = SkimReaderTool(
        api_key="sk402_test_key",
        private_key="0x" + "a" * 64,
    )
    session = tool._get_session()

    assert session is fake
    assert tool._card_lane is True
    assert session.headers["Authorization"] == "Bearer sk402_test_key"


def test_run_card_lane_uses_token_endpoint_get():
    """Card-key reads use the credit endpoint, query params, and timeout."""
    response = _FakeResponse(json_data={"markdown": "card result", "metadata": {}})
    tool = _make_tool(response, api_key="sk402_test_key")
    tool._card_lane = True

    result = tool._run(url="https://example.com/card")

    assert result == "card result"
    call = tool._session.calls[0]
    assert call["method"] == "GET"
    assert call["endpoint"] == "https://skim402.com/api/t/read"
    assert call["params"] == {"url": "https://example.com/card"}
    assert call["timeout"] == 60.0


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
    monkeypatch.delenv("SKIM_API_KEY", raising=False)
    monkeypatch.delenv("SKIM_WALLET_PRIVATE_KEY", raising=False)
    tool = SkimReaderTool()

    with pytest.raises(ValueError, match="SKIM_API_KEY"):
        tool._get_session()


def test_get_session_rejects_malformed_key(monkeypatch):
    monkeypatch.delenv("SKIM_API_KEY", raising=False)
    monkeypatch.setenv("SKIM_WALLET_PRIVATE_KEY", "not-a-valid-hex-key")
    tool = SkimReaderTool()

    with pytest.raises(ValueError, match="64-character hex"):
        tool._get_session()


def test_get_session_builds_wrapped_session(monkeypatch):
    # Mock the lazily-imported x402 / eth_account symbols so no real client,
    # signing, or network is involved.
    calls = {}

    class _FakeModule:
        def __init__(self, attrs):
            self.__dict__.update(attrs)

    def fake_import_module(name):
        if name == "eth_account":
            return _FakeModule(
                {"Account": _FakeModule({"from_key": lambda k: ("account", k)})}
            )
        if name == "x402":
            return _FakeModule({"x402ClientSync": lambda: "client"})
        if name == "x402.client":
            return _FakeModule({"max_amount": lambda cap: ("max_amount", cap)})
        if name == "x402.http.clients.requests":

            def wrap(session, client):
                calls["wrap"] = (session, client)
                return "wrapped-session"

            return _FakeModule({"wrapRequestsWithPayment": wrap})
        if name == "x402.mechanisms.evm.exact.register":

            def register(client, signer, policies=None):
                calls["register"] = (client, signer, policies)

            return _FakeModule({"register_exact_evm_client": register})
        if name == "x402.mechanisms.evm.signers":
            return _FakeModule({"EthAccountSigner": lambda acct: ("signer", acct)})
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(skim_module.importlib, "import_module", fake_import_module)

    tool = SkimReaderTool(private_key="0x" + "a" * 64, max_price_usd=0.01)
    session = tool._get_session()

    assert session == "wrapped-session"
    assert tool._session == "wrapped-session"
    # The payment client was registered and the requests session was wrapped.
    assert "register" in calls
    assert calls["register"][2] == [("max_amount", 10_000)]  # $0.01 -> 10000 atomic
    assert "wrap" in calls
    # A second call reuses the cached session without re-importing.
    monkeypatch.setattr(
        skim_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(AssertionError("should not re-import")),
    )
    assert tool._get_session() == "wrapped-session"


def test_run_validates_url(monkeypatch):
    seen = {}

    def spy_validate(url):
        seen["url"] = url
        return url

    monkeypatch.setattr(skim_module, "validate_url", spy_validate)

    response = _FakeResponse(json_data={"text": "ok", "metadata": {}})
    tool = _make_tool(response)
    tool._run(url="https://example.com/article")

    assert seen["url"] == "https://example.com/article"
