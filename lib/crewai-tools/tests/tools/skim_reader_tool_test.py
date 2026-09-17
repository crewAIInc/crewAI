"""Unit tests for SkimReaderTool.

These tests inject a fake payment-aware session (via the cached ``_session``
attribute), so they never touch the network or sign a real payment.
"""

import sys
from unittest import mock

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


def test_get_session_builds_and_caches_payment_session():
    """_get_session() should select the wallet signer, cap the price in atomic
    USDC units, register the x402 client, and cache the resulting session."""
    fake_account = mock.MagicMock(name="account")
    fake_account_cls = mock.MagicMock()
    fake_account_cls.from_key.return_value = fake_account

    fake_client = mock.MagicMock(name="client")
    fake_client_cls = mock.MagicMock(return_value=fake_client)

    fake_signer = mock.MagicMock(name="signer")
    fake_signer_cls = mock.MagicMock(return_value=fake_signer)

    fake_policy = mock.MagicMock(name="policy")
    fake_max_amount = mock.MagicMock(return_value=fake_policy)

    fake_register = mock.MagicMock(name="register_exact_evm_client")

    fake_wrapped_session = mock.MagicMock(name="wrapped_session")
    fake_wrap = mock.MagicMock(return_value=fake_wrapped_session)

    fake_requests_session = mock.MagicMock(name="requests.Session()")
    fake_requests = mock.MagicMock()
    fake_requests.Session.return_value = fake_requests_session

    fake_modules = {
        "requests": fake_requests,
        "eth_account": mock.MagicMock(Account=fake_account_cls),
        "x402": mock.MagicMock(x402ClientSync=fake_client_cls),
        "x402.client": mock.MagicMock(max_amount=fake_max_amount),
        "x402.http": mock.MagicMock(),
        "x402.http.clients": mock.MagicMock(),
        "x402.http.clients.requests": mock.MagicMock(
            wrapRequestsWithPayment=fake_wrap
        ),
        "x402.mechanisms": mock.MagicMock(),
        "x402.mechanisms.evm": mock.MagicMock(),
        "x402.mechanisms.evm.exact": mock.MagicMock(),
        "x402.mechanisms.evm.exact.register": mock.MagicMock(
            register_exact_evm_client=fake_register
        ),
        "x402.mechanisms.evm.signers": mock.MagicMock(EthAccountSigner=fake_signer_cls),
    }

    tool = SkimReaderTool(private_key=VALID_KEY, max_price_usd=0.05)

    with mock.patch.dict(sys.modules, fake_modules):
        session = tool._get_session()

    fake_account_cls.from_key.assert_called_once_with(VALID_KEY)
    fake_signer_cls.assert_called_once_with(fake_account)
    fake_max_amount.assert_called_once_with(50000)  # 0.05 USD -> 6-decimal USDC
    fake_register.assert_called_once_with(
        fake_client, fake_signer, policies=[fake_policy]
    )
    fake_wrap.assert_called_once_with(fake_requests_session, fake_client)
    assert session is fake_wrapped_session
    assert tool._session is fake_wrapped_session

    # Second call must reuse the cached session, not rebuild it.
    with mock.patch.dict(sys.modules, fake_modules):
        assert tool._get_session() is fake_wrapped_session
    assert fake_wrap.call_count == 1
