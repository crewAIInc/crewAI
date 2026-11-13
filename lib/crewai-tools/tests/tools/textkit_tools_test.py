import json
import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from crewai_tools import (
    TextKitCheckVerificationTool,
    TextKitListMessagesTool,
    TextKitSendMessageTool,
    TextKitSendVerificationTool,
)
from crewai_tools.tools.textkit_tool.textkit_tools import (
    TEXTKIT_API_URL,
    TextKitListMessagesInput,
    TextKitSendMessageInput,
)


REQUEST = "crewai_tools.tools.textkit_tool.textkit_tools.requests.request"


def _response(status_code: int = 200, body: object | None = None) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status_code
    response.ok = status_code < 400
    response.text = json.dumps(body if body is not None else {})
    response.json.return_value = body if body is not None else {}
    return response


@pytest.fixture(autouse=True)
def api_key():
    with patch.dict(os.environ, {"TEXTKIT_API_KEY": "ssms_sk_test_abc"}):
        yield


@patch(REQUEST)
def test_send_message(mock_request):
    mock_request.return_value = _response(201, {"id": "msg_1", "status": "queued"})

    result = TextKitSendMessageTool()._run(
        to="+15005550006", body="Hello", from_number="+15005550100"
    )

    assert json.loads(result)["id"] == "msg_1"
    args, kwargs = mock_request.call_args
    assert args == ("POST", f"{TEXTKIT_API_URL}/messages")
    assert kwargs["headers"] == {"Authorization": "Bearer ssms_sk_test_abc"}
    assert kwargs["json"] == {
        "to": "+15005550006",
        "body": "Hello",
        "from": "+15005550100",
    }
    assert kwargs["timeout"] == 30


@patch(REQUEST)
def test_send_message_omits_unset_from(mock_request):
    mock_request.return_value = _response(201)

    TextKitSendMessageTool()._run(to="+15005550006", body="Hello")

    assert mock_request.call_args.kwargs["json"] == {
        "to": "+15005550006",
        "body": "Hello",
    }


@patch(REQUEST)
def test_send_message_passes_idempotency_key(mock_request):
    mock_request.return_value = _response(201)

    TextKitSendMessageTool()._run(
        to="+15005550006", body="Hello", idempotency_key="order-42"
    )

    assert mock_request.call_args.kwargs["headers"]["Idempotency-Key"] == "order-42"


def test_send_message_rejects_oversized_body():
    with pytest.raises(ValueError):
        TextKitSendMessageInput(to="+15005550006", body="x" * 1601)


@patch(REQUEST)
def test_list_messages_sends_only_set_filters(mock_request):
    mock_request.return_value = _response(
        200, {"data": [], "has_more": False, "next_cursor": None}
    )

    result = TextKitListMessagesTool()._run(limit=5, direction="inbound")

    assert json.loads(result)["has_more"] is False
    args, kwargs = mock_request.call_args
    assert args == ("GET", f"{TEXTKIT_API_URL}/messages")
    assert kwargs["params"] == {"limit": 5, "direction": "inbound"}


def test_list_messages_limit_is_capped():
    with pytest.raises(ValueError):
        TextKitListMessagesInput(limit=101)


@patch(REQUEST)
def test_send_verification(mock_request):
    mock_request.return_value = _response(201, {"id": "ver_1", "status": "pending"})

    result = TextKitSendVerificationTool()._run(phone="+14155550132", app_name="Acme")

    assert json.loads(result)["status"] == "pending"
    args, kwargs = mock_request.call_args
    assert args == ("POST", f"{TEXTKIT_API_URL}/verify")
    assert kwargs["json"] == {"phone": "+14155550132", "app_name": "Acme"}


@patch(REQUEST)
def test_check_verification(mock_request):
    mock_request.return_value = _response(200, {"verified": True})

    result = TextKitCheckVerificationTool()._run(phone="+14155550132", code="482193")

    assert json.loads(result)["verified"] is True
    args, kwargs = mock_request.call_args
    assert args == ("POST", f"{TEXTKIT_API_URL}/verify/check")
    assert kwargs["json"] == {"phone": "+14155550132", "code": "482193"}


def test_missing_api_key_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="TEXTKIT_API_KEY"):
            TextKitSendMessageTool()._run(to="+15005550006", body="Hello")


@patch(REQUEST)
def test_api_error_is_raised_with_code_and_message(mock_request):
    mock_request.return_value = _response(
        402, {"error": {"code": "insufficient_credits", "message": "Add credits."}}
    )

    with pytest.raises(RuntimeError, match="402.*insufficient_credits.*Add credits"):
        TextKitSendMessageTool()._run(to="+15005550006", body="Hello")


@patch(REQUEST)
def test_non_json_error_falls_back_to_text(mock_request):
    response = _response(502)
    response.text = "Bad Gateway"
    response.json.side_effect = ValueError
    mock_request.return_value = response

    with pytest.raises(RuntimeError, match="502.*Bad Gateway"):
        TextKitListMessagesTool()._run()


@patch(REQUEST, side_effect=requests.ConnectionError("boom"))
def test_network_error_is_wrapped(_mock_request):
    with pytest.raises(RuntimeError, match="Could not reach the TextKit API"):
        TextKitListMessagesTool()._run()


@pytest.mark.asyncio
@patch(REQUEST)
async def test_async_run_matches_sync(mock_request):
    mock_request.return_value = _response(200, {"verified": False})

    result = await TextKitCheckVerificationTool()._arun(
        phone="+14155550132", code="000000"
    )

    assert json.loads(result)["verified"] is False
