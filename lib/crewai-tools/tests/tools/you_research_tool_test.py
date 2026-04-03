import json
import os
from unittest.mock import patch

import pytest

from crewai_tools.tools.you_research_tool.you_research_tool import YouResearchTool


@pytest.fixture(autouse=True)
def mock_you_api_key():
    with patch.dict(os.environ, {"YOU_API_KEY": "test_key"}):
        yield


@pytest.fixture
def you_research_tool():
    return YouResearchTool()


def test_you_research_tool_initialization():
    tool = YouResearchTool()
    assert tool.research_effort == "standard"
    assert tool.timeout == 120
    assert tool.research_url == "https://api.you.com/v1/research"


def test_you_research_tool_custom_initialization():
    tool = YouResearchTool(
        research_effort="deep",
        timeout=180,
    )
    assert tool.research_effort == "deep"
    assert tool.timeout == 180


@patch("requests.post")
def test_you_research_tool_basic_research(mock_post, you_research_tool):
    mock_response = {
        "output": {
            "content": "Comprehensive answer with citations [[1]].",
            "content_type": "text",
            "sources": [
                {
                    "url": "https://example.com/article",
                    "title": "Example Article",
                    "snippets": ["Relevant excerpt from the source."],
                }
            ],
        }
    }
    mock_post.return_value.json.return_value = mock_response
    mock_post.return_value.status_code = 200

    result = you_research_tool.run(input="What is quantum computing?")

    assert result is not None
    data = json.loads(result)
    assert "output" in data
    assert "content" in data["output"]
    assert "sources" in data["output"]
    assert len(data["output"]["sources"]) == 1
    assert data["output"]["sources"][0]["url"] == "https://example.com/article"


@patch("requests.post")
def test_you_research_tool_payload(mock_post):
    tool = YouResearchTool()
    mock_post.return_value.json.return_value = {"output": {"content": "", "content_type": "text", "sources": []}}
    mock_post.return_value.status_code = 200

    tool.run(input="test question", research_effort="deep")

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert payload["input"] == "test question"
    assert payload["research_effort"] == "deep"


@patch("requests.post")
def test_you_research_tool_default_effort(mock_post):
    tool = YouResearchTool()
    mock_post.return_value.json.return_value = {"output": {"content": "", "content_type": "text", "sources": []}}
    mock_post.return_value.status_code = 200

    tool.run(input="test question")

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert payload["research_effort"] == "standard"  # falls back to self.research_effort


@patch("requests.post")
def test_you_research_tool_instance_effort_respected(mock_post):
    """Instance-level research_effort is used when not overridden per-call."""
    tool = YouResearchTool(research_effort="exhaustive")
    mock_post.return_value.json.return_value = {"output": {"content": "", "content_type": "text", "sources": []}}
    mock_post.return_value.status_code = 200

    tool.run(input="test question")

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert payload["research_effort"] == "exhaustive"


@patch("requests.post")
def test_you_research_tool_per_call_overrides_instance(mock_post):
    """Per-call research_effort overrides the instance-level default."""
    tool = YouResearchTool(research_effort="exhaustive")
    mock_post.return_value.json.return_value = {"output": {"content": "", "content_type": "text", "sources": []}}
    mock_post.return_value.status_code = 200

    tool.run(input="test question", research_effort="lite")

    call_args = mock_post.call_args
    payload = call_args.kwargs["json"]
    assert payload["research_effort"] == "lite"


@patch("requests.post")
def test_you_research_tool_all_effort_levels(mock_post):
    mock_post.return_value.json.return_value = {"output": {"content": "", "content_type": "text", "sources": []}}
    mock_post.return_value.status_code = 200

    for effort in ["lite", "standard", "deep", "exhaustive"]:
        tool = YouResearchTool()
        tool.run(input="test question", research_effort=effort)

        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["research_effort"] == effort


@patch("requests.post")
def test_you_research_tool_headers(mock_post, you_research_tool):
    mock_post.return_value.json.return_value = {"output": {"content": "", "content_type": "text", "sources": []}}
    mock_post.return_value.status_code = 200

    you_research_tool.run(input="test question")

    call_args = mock_post.call_args
    headers = call_args.kwargs["headers"]
    assert "X-API-Key" in headers
    assert headers["X-API-Key"] == "test_key"
    assert headers["Content-Type"] == "application/json"


@patch("requests.post")
def test_you_research_tool_api_error(mock_post):
    import requests

    tool = YouResearchTool()
    mock_post.side_effect = requests.RequestException("API Error")

    result = tool.run(input="test question")

    assert "Error performing research" in result
    assert "API Error" in result


@patch("requests.post")
def test_you_research_tool_request_exception(mock_post):
    import requests

    tool = YouResearchTool()
    mock_post.side_effect = requests.RequestException("Connection error")

    result = tool.run(input="test question")

    assert "Error performing research" in result
    assert "Connection error" in result


def test_you_research_tool_missing_input():
    tool = YouResearchTool()

    with pytest.raises((TypeError, ValueError)):
        tool.run()


def test_you_research_tool_missing_api_key():
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("YOU_API_KEY", None)
        with pytest.raises(ValueError, match="YOU_API_KEY"):
            YouResearchTool()


if __name__ == "__main__":
    pytest.main([__file__])
