import json
import os
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_valyu_api_key():
    with patch.dict(os.environ, {"VALYU_API_KEY": "test_key_from_env"}):
        yield


@pytest.fixture
def mock_valyu_class():
    with patch(
        "crewai_tools.tools.valyu_extractor_tool.valyu_extractor_tool.Valyu"
    ) as mock_class:
        yield mock_class


@pytest.fixture
def mock_valyu_available():
    with patch(
        "crewai_tools.tools.valyu_extractor_tool.valyu_extractor_tool.VALYU_AVAILABLE",
        True,
    ):
        yield


def test_valyu_extractor_tool_initialization(mock_valyu_class, mock_valyu_available):
    """Test tool initializes correctly with explicit API key."""
    from crewai_tools import ValyuExtractorTool

    api_key = "test_api_key"
    tool = ValyuExtractorTool(api_key=api_key)

    assert tool.api_key == api_key
    assert tool.response_length == "short"
    assert tool.extract_effort == "normal"
    assert tool.screenshot is False
    assert tool.summary is False
    mock_valyu_class.assert_called_once_with(api_key=api_key)


def test_valyu_extractor_tool_initialization_with_env(
    mock_valyu_api_key, mock_valyu_class, mock_valyu_available
):
    """Test tool initializes correctly with API key from environment variable."""
    from crewai_tools import ValyuExtractorTool

    tool = ValyuExtractorTool()
    assert tool.api_key == "test_key_from_env"
    mock_valyu_class.assert_called_once_with(api_key="test_key_from_env")


def test_valyu_extractor_tool_initialization_with_custom_params(
    mock_valyu_class, mock_valyu_available
):
    """Test tool initializes correctly with custom parameters."""
    from crewai_tools import ValyuExtractorTool

    tool = ValyuExtractorTool(
        api_key="test_api_key",
        response_length="large",
        extract_effort="high",
        screenshot=True,
        summary="Summarize the main points",
    )

    assert tool.response_length == "large"
    assert tool.extract_effort == "high"
    assert tool.screenshot is True
    assert tool.summary == "Summarize the main points"


def test_valyu_extractor_tool_run_single_url(mock_valyu_class, mock_valyu_available):
    """Test the _run method with a single URL string."""
    from crewai_tools import ValyuExtractorTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [
            {
                "url": "https://example.com",
                "title": "Example Page",
                "content": "Page content here",
            }
        ]
    }
    mock_valyu_class.return_value.contents.return_value = mock_response

    tool = ValyuExtractorTool(api_key="test_api_key")
    result = tool._run(urls="https://example.com")

    parsed_result = json.loads(result)
    assert "results" in parsed_result
    assert len(parsed_result["results"]) == 1
    assert parsed_result["results"][0]["title"] == "Example Page"

    mock_valyu_class.return_value.contents.assert_called_once_with(
        urls=["https://example.com"],
        response_length="short",
        extract_effort="normal",
        screenshot=False,
    )


def test_valyu_extractor_tool_run_multiple_urls(mock_valyu_class, mock_valyu_available):
    """Test the _run method with multiple URLs."""
    from crewai_tools import ValyuExtractorTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [
            {"url": "https://example1.com", "title": "Page 1", "content": "Content 1"},
            {"url": "https://example2.com", "title": "Page 2", "content": "Content 2"},
        ]
    }
    mock_valyu_class.return_value.contents.return_value = mock_response

    tool = ValyuExtractorTool(api_key="test_api_key")
    result = tool._run(urls=["https://example1.com", "https://example2.com"])

    parsed_result = json.loads(result)
    assert len(parsed_result["results"]) == 2

    mock_valyu_class.return_value.contents.assert_called_once_with(
        urls=["https://example1.com", "https://example2.com"],
        response_length="short",
        extract_effort="normal",
        screenshot=False,
    )


def test_valyu_extractor_tool_run_with_summary_bool(
    mock_valyu_class, mock_valyu_available
):
    """Test the _run method with summary enabled as boolean."""
    from crewai_tools import ValyuExtractorTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [{"url": "https://example.com", "summary": "A summary"}]
    }
    mock_valyu_class.return_value.contents.return_value = mock_response

    tool = ValyuExtractorTool(api_key="test_api_key", summary=True)
    tool._run(urls="https://example.com")

    mock_valyu_class.return_value.contents.assert_called_once_with(
        urls=["https://example.com"],
        response_length="short",
        extract_effort="normal",
        screenshot=False,
        summary=True,
    )


def test_valyu_extractor_tool_run_with_summary_string(
    mock_valyu_class, mock_valyu_available
):
    """Test the _run method with custom summary instructions."""
    from crewai_tools import ValyuExtractorTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"results": []}
    mock_valyu_class.return_value.contents.return_value = mock_response

    custom_summary = "Focus on technical details"
    tool = ValyuExtractorTool(api_key="test_api_key", summary=custom_summary)
    tool._run(urls="https://example.com")

    mock_valyu_class.return_value.contents.assert_called_once_with(
        urls=["https://example.com"],
        response_length="short",
        extract_effort="normal",
        screenshot=False,
        summary=custom_summary,
    )


def test_valyu_extractor_tool_run_with_screenshot(
    mock_valyu_class, mock_valyu_available
):
    """Test the _run method with screenshot enabled."""
    from crewai_tools import ValyuExtractorTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [{"url": "https://example.com", "screenshot_url": "https://..."}]
    }
    mock_valyu_class.return_value.contents.return_value = mock_response

    tool = ValyuExtractorTool(api_key="test_api_key", screenshot=True)
    tool._run(urls="https://example.com")

    mock_valyu_class.return_value.contents.assert_called_once_with(
        urls=["https://example.com"],
        response_length="short",
        extract_effort="normal",
        screenshot=True,
    )


def test_valyu_extractor_tool_run_no_client_raises_error(
    mock_valyu_class, mock_valyu_available
):
    """Test that _run raises error when client is not initialized."""
    from crewai_tools import ValyuExtractorTool

    tool = ValyuExtractorTool(api_key="test_api_key")
    tool.client = None

    with pytest.raises(ValueError, match="Valyu client is not initialized"):
        tool._run(urls="https://example.com")


@pytest.mark.asyncio
async def test_valyu_extractor_tool_arun(mock_valyu_class, mock_valyu_available):
    """Test the async _arun method."""
    from crewai_tools import ValyuExtractorTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [{"url": "https://example.com", "content": "Content"}]
    }
    mock_valyu_class.return_value.contents.return_value = mock_response

    tool = ValyuExtractorTool(api_key="test_api_key")
    result = await tool._arun(urls="https://example.com")

    parsed_result = json.loads(result)
    assert "results" in parsed_result


@pytest.mark.asyncio
async def test_valyu_extractor_tool_arun_multiple_urls(
    mock_valyu_class, mock_valyu_available
):
    """Test the async _arun method with multiple URLs."""
    from crewai_tools import ValyuExtractorTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [
            {"url": "https://example1.com"},
            {"url": "https://example2.com"},
        ]
    }
    mock_valyu_class.return_value.contents.return_value = mock_response

    tool = ValyuExtractorTool(api_key="test_api_key")
    result = await tool._arun(urls=["https://example1.com", "https://example2.com"])

    parsed_result = json.loads(result)
    assert len(parsed_result["results"]) == 2


def test_valyu_extractor_tool_handles_dict_response(
    mock_valyu_class, mock_valyu_available
):
    """Test that tool handles dict response (legacy .dict() method)."""
    from crewai_tools import ValyuExtractorTool

    mock_response = MagicMock(spec=[])
    mock_response.dict = MagicMock(
        return_value={"results": [{"url": "https://example.com", "content": "Content"}]}
    )
    del mock_response.model_dump
    mock_valyu_class.return_value.contents.return_value = mock_response

    tool = ValyuExtractorTool(api_key="test_api_key")
    result = tool._run(urls="https://example.com")

    parsed_result = json.loads(result)
    assert "results" in parsed_result
    mock_response.dict.assert_called_once()


def test_valyu_extractor_tool_handles_raw_dict_response(
    mock_valyu_class, mock_valyu_available
):
    """Test that tool handles raw dict response."""
    from crewai_tools import ValyuExtractorTool

    mock_valyu_class.return_value.contents.return_value = {
        "results": [{"url": "https://example.com", "content": "Content"}]
    }

    tool = ValyuExtractorTool(api_key="test_api_key")
    result = tool._run(urls="https://example.com")

    parsed_result = json.loads(result)
    assert "results" in parsed_result


def test_valyu_extractor_tool_name_and_description(
    mock_valyu_class, mock_valyu_available
):
    """Test tool has correct name and description."""
    from crewai_tools import ValyuExtractorTool

    tool = ValyuExtractorTool(api_key="test_api_key")

    assert tool.name == "Valyu Extractor"
    assert "Extracts clean, structured content" in tool.description
    assert "web pages" in tool.description


def test_valyu_extractor_tool_env_vars_config(mock_valyu_class, mock_valyu_available):
    """Test tool has correct environment variable configuration."""
    from crewai_tools import ValyuExtractorTool

    tool = ValyuExtractorTool(api_key="test_api_key")

    assert len(tool.env_vars) == 1
    assert tool.env_vars[0].name == "VALYU_API_KEY"
    assert tool.env_vars[0].required is True


def test_valyu_extractor_tool_package_dependencies(
    mock_valyu_class, mock_valyu_available
):
    """Test tool has correct package dependencies."""
    from crewai_tools import ValyuExtractorTool

    tool = ValyuExtractorTool(api_key="test_api_key")

    assert "valyu" in tool.package_dependencies


def test_valyu_extractor_tool_all_response_lengths(
    mock_valyu_class, mock_valyu_available
):
    """Test tool accepts all valid response_length values."""
    from crewai_tools import ValyuExtractorTool

    for length in ["short", "medium", "large", "max"]:
        tool = ValyuExtractorTool(api_key="test_api_key", response_length=length)
        assert tool.response_length == length


def test_valyu_extractor_tool_all_extract_efforts(
    mock_valyu_class, mock_valyu_available
):
    """Test tool accepts all valid extract_effort values."""
    from crewai_tools import ValyuExtractorTool

    for effort in ["normal", "high", "auto"]:
        tool = ValyuExtractorTool(api_key="test_api_key", extract_effort=effort)
        assert tool.extract_effort == effort
