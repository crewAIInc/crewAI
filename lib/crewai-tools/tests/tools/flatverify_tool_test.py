import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

from crewai.tools.tool_failure import ToolFailure
from crewai_tools import FlatverifyBulkVerifyTool, FlatverifyVerifyEmailTool
import pytest
import requests


TEST_KEY = "fv_test_0123456789abcdef"
JOB_ID = "8f14e45f-ceea-4e7a-9d3b-2f1c6a1b7e21"
VALID = {
    "email": "maya.chen@amazon.com",
    "status": "valid",
    "sub_status": "accepted",
    "reason": "The address can receive mail.",
    "disposable": False,
    "role": False,
    "free": False,
    "catch_all": False,
}
ROWS = [
    {**VALID, "email": "j.ortiz@salesforce.com"},
    {
        **VALID,
        "email": "sam.lee@stripe.com",
        "status": "invalid",
        "sub_status": "mailbox_not_found",
    },
    {
        **VALID,
        "email": "info@shopify.com",
        "status": "risky",
        "sub_status": "role_account",
        "role": True,
    },
]


def _job(status: str = "done") -> dict[str, Any]:
    return {
        "id": JOB_ID,
        "name": None,
        "source": "api",
        "status": status,
        "waiting_reason": None,
        "total": 3,
        "processed": 3 if status == "done" else 1,
        "counts": {"valid": 1, "invalid": 1, "risky": 1, "unknown": 0},
        "created_at": "2026-11-22T10:00:00.000Z",
        "finished_at": None,
    }


def _response(
    status: int, body: dict[str, Any], headers: dict[str, str] | None = None
) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp._content = json.dumps(body).encode()
    resp.headers.update(headers or {})
    return resp


def _error(status: int, code: str, retry_after: str | None = None) -> requests.Response:
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return _response(status, {"error": {"code": code, "message": f"{code} message"}}, headers)


@pytest.fixture(autouse=True)
def api_key():
    with patch.dict(os.environ, {"FLATVERIFY_API_KEY": TEST_KEY}):
        yield


@pytest.fixture
def http():
    with patch.object(requests.Session, "request") as mock_request:
        yield mock_request


def test_requires_api_key():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="FLATVERIFY_API_KEY"):
            FlatverifyVerifyEmailTool()


def test_api_key_argument_and_env_var_declared():
    tool = FlatverifyVerifyEmailTool(api_key="fv_other")
    assert [v.name for v in tool.env_vars] == ["FLATVERIFY_API_KEY"]
    assert "fv_other" not in repr(tool)
    assert TEST_KEY not in repr(FlatverifyBulkVerifyTool())


def test_verify_email(http: MagicMock):
    http.return_value = _response(200, VALID)
    tool = FlatverifyVerifyEmailTool()
    assert tool.run(email="maya.chen@amazon.com") == VALID
    method, url = http.call_args.args
    assert (method, url) == ("POST", "https://app.flatverify.com/api/v1/verify")
    assert http.call_args.kwargs["json"] == {"email": "maya.chen@amazon.com"}
    assert tool._client._session.headers["Authorization"] == f"Bearer {TEST_KEY}"


def test_retries_rate_limited_and_busy(http: MagicMock):
    http.side_effect = [
        _error(429, "rate_limited", "0"),
        _error(503, "busy", "0"),
        _response(200, VALID),
    ]
    assert FlatverifyVerifyEmailTool().run(email="maya.chen@amazon.com") == VALID
    assert http.call_count == 3


def test_retry_wait_is_capped(http: MagicMock):
    http.side_effect = [_error(429, "rate_limited", "3600"), _response(200, VALID)]
    with patch(
        "crewai_tools.tools.flatverify_tool.flatverify_tool.time.sleep"
    ) as sleep:
        FlatverifyVerifyEmailTool(max_retry_wait=5).run(email="maya.chen@amazon.com")
    sleep.assert_called_once_with(5.0)


def test_gives_up_after_max_retries(http: MagicMock):
    http.return_value = _error(503, "busy", "0")
    result = FlatverifyVerifyEmailTool(max_retries=2).run(email="maya.chen@amazon.com")
    assert isinstance(result, ToolFailure)
    assert result.code == "busy"
    assert result.retryable is True
    assert http.call_count == 3


def test_client_errors_are_not_retried(http: MagicMock):
    http.return_value = _error(401, "unauthorized")
    result = FlatverifyVerifyEmailTool().run(email="maya.chen@amazon.com")
    assert isinstance(result, ToolFailure)
    assert result.code == "unauthorized"
    assert result.retryable is False
    assert TEST_KEY not in result.message
    assert http.call_count == 1


def test_network_error_is_reported(http: MagicMock):
    http.side_effect = requests.ConnectionError("down")
    result = FlatverifyVerifyEmailTool().run(email="maya.chen@amazon.com")
    assert isinstance(result, ToolFailure)
    assert "Could not reach" in result.message


def test_bulk_waits_and_returns_rows(http: MagicMock):
    http.side_effect = [
        _response(202, {**_job("queued"), "skipped": 1}),
        _response(200, _job("running")),
        _response(200, _job("done")),
        _response(200, {"results": ROWS[:2], "next_after": 1}),
        _response(200, {"results": ROWS[2:], "next_after": None}),
    ]
    tool = FlatverifyBulkVerifyTool(poll_interval=0)
    emails = [row["email"] for row in ROWS]
    result = tool.run(emails=emails, name="Q4 leads")
    assert http.call_args_list[0].kwargs["json"] == {"emails": emails, "name": "Q4 leads"}
    assert http.call_args_list[4].kwargs["params"] == {"format": "json", "after": 1}
    assert result["job_id"] == JOB_ID
    assert result["status"] == "done"
    assert result["skipped"] == 1
    assert result["counts"]["invalid"] == 1
    assert result["results"] == ROWS
    assert result["results_truncated"] is False


def test_bulk_truncates_rows(http: MagicMock):
    http.side_effect = [
        _response(202, {**_job("queued"), "skipped": 0}),
        _response(200, _job("done")),
        _response(200, {"results": ROWS, "next_after": None}),
    ]
    result = FlatverifyBulkVerifyTool(max_results=2).run(emails=["j.ortiz@salesforce.com"])
    assert len(result["results"]) == 2
    assert result["results_truncated"] is True


def test_bulk_returns_job_id_when_wait_ends_then_resumes(http: MagicMock):
    http.side_effect = [
        _response(202, {**_job("queued"), "skipped": 0}),
        _response(200, _job("running")),
    ]
    tool = FlatverifyBulkVerifyTool(wait_timeout=0)
    first = tool.run(emails=["j.ortiz@salesforce.com"])
    assert first["status"] == "running"
    assert first["job_id"] == JOB_ID
    assert "results" not in first
    http.side_effect = [
        _response(200, _job("done")),
        _response(200, {"results": ROWS, "next_after": None}),
    ]
    second = tool.run(job_id=JOB_ID)
    assert second["results"] == ROWS
    assert http.call_args_list[2].args == (
        "GET",
        f"https://app.flatverify.com/api/v1/jobs/{JOB_ID}",
    )


def test_bulk_too_many_jobs_is_not_retried(http: MagicMock):
    http.return_value = _error(429, "too_many_jobs", "0")
    result = FlatverifyBulkVerifyTool().run(emails=["j.ortiz@salesforce.com"])
    assert isinstance(result, ToolFailure)
    assert result.code == "too_many_jobs"
    assert http.call_count == 1


def test_bulk_needs_emails_or_job_id(http: MagicMock):
    result = FlatverifyBulkVerifyTool().run(emails=[])
    assert isinstance(result, ToolFailure)
    assert result.code == "invalid_request"
    http.assert_not_called()
