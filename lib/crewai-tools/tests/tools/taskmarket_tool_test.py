"""Tests for TaskMarketTool."""

import json
from unittest.mock import Mock

import pytest
import requests

from crewai_tools import TaskMarketTool

TASK_ID = "0x" + "a" * 64


def response(payload: object, status: int = 200) -> Mock:
    """Build a minimal streaming response mock."""
    result = Mock()
    result.status_code = status
    result.iter_content.return_value = [json.dumps(payload).encode()]
    if status >= 400:
        error = requests.HTTPError()
        error.response = result
        result.raise_for_status.side_effect = error
    return result


def test_list_tasks_builds_bounded_filters():
    """Whole-USDC filters are converted to TaskMarket's six-decimal base units."""
    session = Mock()
    session.get.return_value = response({"tasks": [], "hasMore": False})
    tool = TaskMarketTool(session=session)

    result = json.loads(
        tool.run(
            operation="list_tasks",
            status="open",
            mode="bounty",
            tags=["python", "agents"],
            min_reward_usdc=0.5,
            max_reward_usdc=2,
            limit=10,
        )
    )

    assert result["success"] is True
    session.get.assert_called_once_with(
        "https://api.taskmarket.dev/api/tasks",
        params={
            "status": "open",
            "mode": "bounty",
            "tags": "python,agents",
            "minReward": "500000",
            "maxReward": "2000000",
            "limit": "10",
        },
        headers={"Accept": "application/json"},
        timeout=10.0,
        stream=True,
    )
    session.get.return_value.close.assert_called_once()


def test_get_task_rejects_malformed_id_before_network():
    """A malformed ID never reaches URL construction or the HTTP session."""
    session = Mock()
    tool = TaskMarketTool(session=session)

    with pytest.raises(ValueError, match="exactly 64 hexadecimal"):
        tool.run(operation="get_task", task_id="../../wallet")

    session.get.assert_not_called()


def test_get_task_returns_public_payload():
    """A public task is returned in a stable JSON envelope."""
    session = Mock()
    session.get.return_value = response({"id": TASK_ID, "status": "open"})

    result = json.loads(TaskMarketTool(session=session).run(operation="get_task", task_id=TASK_ID))

    assert result == {"success": True, "data": {"id": TASK_ID, "status": "open"}}


def test_list_submissions_is_read_only():
    """Expose public submission metadata without write or payment methods."""
    session = Mock()
    session.get.return_value = response([{"id": "submission-1"}])
    tool = TaskMarketTool(session=session)

    result = json.loads(tool.run(operation="list_submissions", task_id=TASK_ID))

    assert result["data"] == [{"id": "submission-1"}]
    assert not hasattr(tool, "create_task")
    assert not hasattr(tool, "accept_submission")
    assert not hasattr(tool, "pay")


def test_http_error_does_not_echo_untrusted_body():
    """Report only the status code from a failing API response."""
    session = Mock()
    session.get.return_value = response({"private": "server body"}, status=403)

    result = json.loads(TaskMarketTool(session=session).run(operation="get_task", task_id=TASK_ID))

    assert result == {"success": False, "error": "TaskMarket API returned HTTP 403"}
    assert "server body" not in json.dumps(result)


def test_timeout_is_stable_and_redacted():
    """Transport details are not returned to an agent on timeout."""
    session = Mock()
    session.get.side_effect = requests.Timeout("internal details")

    result = json.loads(TaskMarketTool(session=session).run(operation="get_task", task_id=TASK_ID))

    assert result == {"success": False, "error": "TaskMarket request timed out"}


def test_response_size_limit_stops_stream():
    """Oversized responses fail and close instead of entering agent context."""
    session = Mock()
    oversized = response([])
    oversized.iter_content.return_value = [b"x" * 1025]
    session.get.return_value = oversized
    tool = TaskMarketTool(session=session, max_response_bytes=1024)

    result = json.loads(tool.run(operation="list_tasks"))

    assert result == {
        "success": False,
        "error": "TaskMarket response exceeded the configured size limit",
    }
    oversized.close.assert_called_once()


def test_invalid_json_is_reported_without_body():
    """Invalid JSON becomes a stable error rather than a traceback."""
    session = Mock()
    invalid = response([])
    invalid.iter_content.return_value = [b"not-json"]
    session.get.return_value = invalid

    result = json.loads(TaskMarketTool(session=session).run(operation="list_tasks"))

    assert result == {"success": False, "error": "TaskMarket returned invalid JSON"}


def test_configuration_and_filter_validation():
    """Reject plaintext origins, contradictory ranges, and delimiter-bearing tags."""
    with pytest.raises(ValueError, match="must use HTTPS"):
        TaskMarketTool(api_url="http://api.taskmarket.dev")

    tool = TaskMarketTool(session=Mock())
    with pytest.raises(ValueError, match="may not exceed"):
        tool.run(operation="list_tasks", min_reward_usdc=2, max_reward_usdc=1)
    with pytest.raises(ValueError, match="may not contain commas"):
        tool.run(operation="list_tasks", tags=["python,open"])
