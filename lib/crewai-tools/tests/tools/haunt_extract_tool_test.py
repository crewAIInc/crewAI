import json
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.tools.haunt_extract_tool.haunt_extract_tool import (
    HauntExtractTool,
)


def _response(status_code, payload):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload
    return response


def test_haunt_extract_success_returns_json():
    tool = HauntExtractTool(api_key="haunt_test_key")
    with patch(
        "crewai_tools.tools.haunt_extract_tool.haunt_extract_tool.requests.post",
        return_value=_response(
            200, {"success": True, "data": {"title": "Example Domain"}}
        ),
    ) as mock_post:
        result = tool.run(url="https://example.com", prompt="the page title")

    assert json.loads(result) == {"title": "Example Domain"}
    body = mock_post.call_args.kwargs["json"]
    assert body == {"url": "https://example.com", "prompt": "the page title"}


def test_haunt_extract_markdown_unwrapped():
    tool = HauntExtractTool(api_key="haunt_test_key", response_format="markdown")
    with patch(
        "crewai_tools.tools.haunt_extract_tool.haunt_extract_tool.requests.post",
        return_value=_response(
            200, {"success": True, "data": {"markdown": "# Example"}}
        ),
    ):
        result = tool.run(url="https://example.com", prompt="the content")

    assert result == "# Example"


def test_haunt_extract_honest_failure_returns_error_code():
    tool = HauntExtractTool(api_key="haunt_test_key")
    with patch(
        "crewai_tools.tools.haunt_extract_tool.haunt_extract_tool.requests.post",
        return_value=_response(
            200,
            {
                "success": False,
                "error_code": "login_required",
                "message": "This page needs a login.",
            },
        ),
    ):
        result = json.loads(
            tool.run(url="https://example.com/admin", prompt="the numbers")
        )

    assert result["error_code"] == "login_required"


def test_haunt_extract_invalid_key_raises():
    tool = HauntExtractTool(api_key="haunt_bad_key")
    with patch(
        "crewai_tools.tools.haunt_extract_tool.haunt_extract_tool.requests.post",
        return_value=_response(401, {"error": "Invalid API key"}),
    ):
        with pytest.raises(ValueError, match="Invalid API key"):
            tool.run(url="https://example.com", prompt="the page title")


def test_haunt_extract_missing_key_raises(monkeypatch):
    monkeypatch.delenv("HAUNT_API_KEY", raising=False)
    tool = HauntExtractTool()
    with pytest.raises(ValueError, match="HAUNT_API_KEY"):
        tool.run(url="https://example.com", prompt="the page title")
