import asyncio
import os
import uuid
from unittest.mock import ANY, MagicMock, patch

import pytest

from crewai_tools.tools.slack_tool.slack_tool import (
    SlackChannelHistoryTool,
    SlackSendMessageTool,
)


@pytest.fixture(autouse=True)
def mock_slack_bot_token():
    with patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test-token"}):
        yield


def _mock_web_client(return_value=None, side_effect=None):
    client = MagicMock()
    if side_effect is not None:
        client.chat_postMessage.side_effect = side_effect
        client.conversations_history.side_effect = side_effect
    else:
        client.chat_postMessage.return_value = return_value
        client.conversations_history.return_value = return_value
    return client


def test_missing_token_raises():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError):
            SlackSendMessageTool()
        with pytest.raises(ValueError):
            SlackChannelHistoryTool()


@patch("crewai_tools.tools.slack_tool.slack_tool._require_slack_sdk")
def test_send_message_happy_path(mock_require_sdk):
    mock_client_cls = MagicMock()
    mock_client_cls.return_value = _mock_web_client(
        return_value={"ok": True, "ts": "1234567890.000100"}
    )
    mock_require_sdk.return_value = mock_client_cls

    tool = SlackSendMessageTool()
    result = tool._run(channel="#general", message="hello world")

    mock_client_cls.return_value.chat_postMessage.assert_called_once_with(
        channel="#general", text="hello world", thread_ts=None, client_msg_id=ANY
    )
    assert "Message posted to #general" in result
    assert "1234567890.000100" in result


@patch("crewai_tools.tools.slack_tool.slack_tool._require_slack_sdk")
def test_send_message_thread_reply(mock_require_sdk):
    mock_client_cls = MagicMock()
    mock_client_cls.return_value = _mock_web_client(
        return_value={"ok": True, "ts": "1234567890.000200"}
    )
    mock_require_sdk.return_value = mock_client_cls

    tool = SlackSendMessageTool()
    tool._run(channel="C0123456789", message="reply", thread_ts="1234567890.000100")

    mock_client_cls.return_value.chat_postMessage.assert_called_once_with(
        channel="C0123456789",
        text="reply",
        thread_ts="1234567890.000100",
        client_msg_id=ANY,
    )


@patch("crewai_tools.tools.slack_tool.slack_tool._require_slack_sdk")
def test_send_message_client_msg_id_is_unique_uuid(mock_require_sdk):
    mock_client_cls = MagicMock()
    mock_client_cls.return_value = _mock_web_client(
        return_value={"ok": True, "ts": "1234567890.000300"}
    )
    mock_require_sdk.return_value = mock_client_cls

    tool = SlackSendMessageTool()
    tool._run(channel="#general", message="first")
    tool._run(channel="#general", message="second")

    calls = mock_client_cls.return_value.chat_postMessage.call_args_list
    first_id = calls[0].kwargs["client_msg_id"]
    second_id = calls[1].kwargs["client_msg_id"]

    assert uuid.UUID(first_id)
    assert uuid.UUID(second_id)
    assert first_id != second_id


@patch("crewai_tools.tools.slack_tool.slack_tool._require_slack_sdk")
def test_send_message_api_error(mock_require_sdk):
    from slack_sdk.errors import SlackApiError

    error_response = MagicMock()
    error_response.get.return_value = "channel_not_found"
    api_error = SlackApiError("error", response=error_response)

    mock_client_cls = MagicMock()
    mock_client_cls.return_value = _mock_web_client(side_effect=api_error)
    mock_require_sdk.return_value = mock_client_cls

    tool = SlackSendMessageTool()
    result = tool._run(channel="#nonexistent", message="hello")

    assert "channel_not_found" in result


@patch("crewai_tools.tools.slack_tool.slack_tool._require_slack_sdk")
def test_channel_history_happy_path(mock_require_sdk):
    mock_client_cls = MagicMock()
    mock_client_cls.return_value = _mock_web_client(
        return_value={
            "ok": True,
            "messages": [
                {"user": "U123", "text": "hi there", "ts": "1234567890.000100"},
                {"user": "U456", "text": "hello!", "ts": "1234567890.000050"},
            ],
        }
    )
    mock_require_sdk.return_value = mock_client_cls

    tool = SlackChannelHistoryTool()
    result = tool._run(channel="C0123456789", limit=20)

    mock_client_cls.return_value.conversations_history.assert_called_once_with(
        channel="C0123456789", limit=20
    )
    assert "hi there" in result
    assert "hello!" in result


@patch("crewai_tools.tools.slack_tool.slack_tool._require_slack_sdk")
def test_channel_history_empty(mock_require_sdk):
    mock_client_cls = MagicMock()
    mock_client_cls.return_value = _mock_web_client(
        return_value={"ok": True, "messages": []}
    )
    mock_require_sdk.return_value = mock_client_cls

    tool = SlackChannelHistoryTool()
    result = tool._run(channel="C0123456789")

    assert "No messages found" in result


@patch("crewai_tools.tools.slack_tool.slack_tool._require_slack_sdk")
def test_channel_history_api_error(mock_require_sdk):
    from slack_sdk.errors import SlackApiError

    error_response = MagicMock()
    error_response.get.return_value = "not_in_channel"
    api_error = SlackApiError("error", response=error_response)

    mock_client_cls = MagicMock()
    mock_client_cls.return_value = _mock_web_client(side_effect=api_error)
    mock_require_sdk.return_value = mock_client_cls

    tool = SlackChannelHistoryTool()
    result = tool._run(channel="C0123456789")

    assert "not_in_channel" in result


def test_history_limit_validation():
    tool = SlackChannelHistoryTool()
    with pytest.raises(Exception):
        tool.run(channel="C0123456789", limit=500)


@pytest.mark.asyncio
@patch("crewai_tools.tools.slack_tool.slack_tool._require_slack_sdk")
async def test_send_message_arun_does_not_block_event_loop(mock_require_sdk):
    mock_client_cls = MagicMock()
    mock_client_cls.return_value = _mock_web_client(
        return_value={"ok": True, "ts": "1234567890.000400"}
    )
    mock_require_sdk.return_value = mock_client_cls

    tool = SlackSendMessageTool()
    heartbeat_ticks = 0

    async def heartbeat() -> None:
        nonlocal heartbeat_ticks
        for _ in range(5):
            await asyncio.sleep(0)
            heartbeat_ticks += 1

    result, _ = await asyncio.gather(
        tool._arun(channel="#general", message="hello async"),
        heartbeat(),
    )

    assert "Message posted to #general" in result
    assert heartbeat_ticks == 5


@pytest.mark.asyncio
@patch("crewai_tools.tools.slack_tool.slack_tool._require_slack_sdk")
async def test_channel_history_arun_delegates_to_run(mock_require_sdk):
    mock_client_cls = MagicMock()
    mock_client_cls.return_value = _mock_web_client(
        return_value={
            "ok": True,
            "messages": [{"user": "U123", "text": "hi", "ts": "1234567890.000500"}],
        }
    )
    mock_require_sdk.return_value = mock_client_cls

    tool = SlackChannelHistoryTool()
    result = await tool._arun(channel="C0123456789", limit=5)

    mock_client_cls.return_value.conversations_history.assert_called_once_with(
        channel="C0123456789", limit=5
    )
    assert "hi" in result
