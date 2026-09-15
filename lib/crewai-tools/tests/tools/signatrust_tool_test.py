import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.signatrust_tool.signatrust_tool import SignatrustTool


@pytest.fixture
def signatrust_tool():
    return SignatrustTool(api_key="sk_test_123")


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("SIGNATRUST_API_KEY", raising=False)
    with pytest.raises(ValueError):
        SignatrustTool()


def test_initialization_with_env(monkeypatch):
    monkeypatch.setenv("SIGNATRUST_API_KEY", "sk_env_456")
    tool = SignatrustTool()
    assert tool.api_key == "sk_env_456"
    assert tool.base_url == "https://signatrust.net/api/v1"


def test_custom_base_url():
    tool = SignatrustTool(api_key="sk_test_123", base_url="https://self.hosted/api/v1")
    assert tool.base_url == "https://self.hosted/api/v1"


@patch("requests.post")
def test_generate(mock_post, signatrust_tool):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id": "rcpt_1", "status": "signed"}
    mock_resp.raise_for_status.return_value = None
    mock_post.return_value = mock_resp

    out = signatrust_tool.run(
        operation="generate",
        agent_name="bot",
        action="approve",
        decision="approved",
        model="gpt-4o",
        human_review=True,
    )
    data = json.loads(out)
    assert data["id"] == "rcpt_1"
    assert data["status"] == "signed"
    # API key passed via header
    _, kwargs = mock_post.call_args
    assert kwargs["headers"]["X-API-Key"] == "sk_test_123"
    # None fields stripped from body
    assert "metadata" not in kwargs["json"]


@patch("requests.get")
def test_verify(mock_get, signatrust_tool):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id": "rcpt_1", "valid": True}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    out = signatrust_tool.run(operation="verify", receipt_id="rcpt_1")
    data = json.loads(out)
    assert data["valid"] is True
    assert mock_get.call_args[0][0].endswith("/receipts/rcpt_1/verify")


@patch("requests.get")
def test_get(mock_get, signatrust_tool):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"id": "rcpt_1", "action": "approve"}
    mock_resp.raise_for_status.return_value = None
    mock_get.return_value = mock_resp

    out = signatrust_tool.run(operation="get", receipt_id="rcpt_1")
    data = json.loads(out)
    assert data["id"] == "rcpt_1"
    assert mock_get.call_args[0][0].endswith("/receipts/rcpt_1")


def test_verify_requires_receipt_id(signatrust_tool):
    out = signatrust_tool.run(operation="verify")
    assert "receipt_id" in out


def test_unknown_operation(signatrust_tool):
    out = signatrust_tool.run(operation="frobnicate")
    assert "unknown operation" in out.lower()


@patch("requests.post")
def test_arun_non_blocking(mock_post, signatrust_tool):
    """`_arun` must offload the blocking HTTP call to a worker thread."""
    main_thread_id = threading.get_ident()
    call_thread_ids: list[int] = []

    def fake_post(*args, **kwargs):
        call_thread_ids.append(threading.get_ident())
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"id": "rcpt_async", "status": "signed"}
        mock_resp.raise_for_status.return_value = None
        return mock_resp

    mock_post.side_effect = fake_post

    out = asyncio.run(
        signatrust_tool._arun(
            operation="generate",
            agent_name="bot",
            action="approve",
            decision="approved",
        )
    )
    data = json.loads(out)
    assert data["id"] == "rcpt_async"
    assert data["status"] == "signed"
    assert len(call_thread_ids) == 1, "requests.post should be invoked exactly once"
    assert call_thread_ids[0] != main_thread_id, (
        "_arun must offload the blocking HTTP call to a worker thread; "
        "got call on the main thread (event loop would block)."
    )


@patch("requests.get")
def test_serialized_output_is_deterministic(mock_get, signatrust_tool):
    """Equivalent payloads must serialize to the same JSON string regardless of upstream key order."""
    payload_keys_order_a = {"id": "rcpt_1", "valid": True, "trust": 90}
    payload_keys_order_b = {"trust": 90, "valid": True, "id": "rcpt_1"}

    outputs = []
    for payload in (payload_keys_order_a, payload_keys_order_b):
        mock_resp = MagicMock()
        mock_resp.json.return_value = payload
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp
        outputs.append(signatrust_tool.run(operation="verify", receipt_id="rcpt_1"))

    assert outputs[0] == outputs[1]
    # And keys are sorted alphabetically.
    assert outputs[0] == '{"id": "rcpt_1", "trust": 90, "valid": true}'
