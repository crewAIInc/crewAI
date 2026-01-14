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
        "crewai_tools.tools.valyu_search_tool.valyu_search_tool.Valyu"
    ) as mock_class:
        yield mock_class


@pytest.fixture
def mock_valyu_available():
    with patch(
        "crewai_tools.tools.valyu_search_tool.valyu_search_tool.VALYU_AVAILABLE", True
    ):
        yield


def test_valyu_search_tool_initialization(mock_valyu_class, mock_valyu_available):
    """Test tool initializes correctly with explicit API key."""
    from crewai_tools import ValyuSearchTool

    api_key = "test_api_key"
    tool = ValyuSearchTool(api_key=api_key)

    assert tool.api_key == api_key
    assert tool.search_type == "all"
    assert tool.max_num_results == 10
    assert tool.relevance_threshold == 0.5
    assert tool.response_length == "short"
    assert tool.max_content_length_per_result == 1000
    assert tool.included_sources is None
    assert tool.excluded_sources is None
    assert tool.start_date is None
    assert tool.end_date is None
    assert tool.country_code is None
    mock_valyu_class.assert_called_once_with(api_key=api_key)


def test_valyu_search_tool_initialization_with_env(
    mock_valyu_api_key, mock_valyu_class, mock_valyu_available
):
    """Test tool initializes correctly with API key from environment variable."""
    from crewai_tools import ValyuSearchTool

    tool = ValyuSearchTool()
    assert tool.api_key == "test_key_from_env"
    mock_valyu_class.assert_called_once_with(api_key="test_key_from_env")


def test_valyu_search_tool_initialization_with_custom_params(
    mock_valyu_class, mock_valyu_available
):
    """Test tool initializes correctly with custom parameters."""
    from crewai_tools import ValyuSearchTool

    tool = ValyuSearchTool(
        api_key="test_api_key",
        search_type="web",
        max_num_results=5,
        relevance_threshold=0.8,
        response_length="large",
        included_sources=["example.com", "test.com"],
        excluded_sources=["spam.com"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        country_code="US",
        max_content_length_per_result=500,
    )

    assert tool.search_type == "web"
    assert tool.max_num_results == 5
    assert tool.relevance_threshold == 0.8
    assert tool.response_length == "large"
    assert list(tool.included_sources) == ["example.com", "test.com"]
    assert list(tool.excluded_sources) == ["spam.com"]
    assert tool.start_date == "2024-01-01"
    assert tool.end_date == "2024-12-31"
    assert tool.country_code == "US"
    assert tool.max_content_length_per_result == 500


def test_valyu_search_tool_run(mock_valyu_class, mock_valyu_available):
    """Test the _run method executes search correctly."""
    from crewai_tools import ValyuSearchTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com",
                "content": "Test content",
                "relevance_score": 0.9,
            }
        ]
    }
    mock_valyu_class.return_value.search.return_value = mock_response

    tool = ValyuSearchTool(api_key="test_api_key")
    result = tool._run(query="test query")

    parsed_result = json.loads(result)
    assert "results" in parsed_result
    assert len(parsed_result["results"]) == 1
    assert parsed_result["results"][0]["title"] == "Test Result"

    mock_valyu_class.return_value.search.assert_called_once_with(
        query="test query",
        search_type="all",
        max_num_results=10,
        relevance_threshold=0.5,
        response_length="short",
    )


def test_valyu_search_tool_run_with_optional_params(
    mock_valyu_class, mock_valyu_available
):
    """Test the _run method includes optional parameters when set."""
    from crewai_tools import ValyuSearchTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"results": []}
    mock_valyu_class.return_value.search.return_value = mock_response

    tool = ValyuSearchTool(
        api_key="test_api_key",
        included_sources=["example.com"],
        excluded_sources=["spam.com"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        country_code="US",
    )
    tool._run(query="test query")

    mock_valyu_class.return_value.search.assert_called_once_with(
        query="test query",
        search_type="all",
        max_num_results=10,
        relevance_threshold=0.5,
        response_length="short",
        included_sources=["example.com"],
        excluded_sources=["spam.com"],
        start_date="2024-01-01",
        end_date="2024-12-31",
        country_code="US",
    )


def test_valyu_search_tool_content_truncation(mock_valyu_class, mock_valyu_available):
    """Test that content is truncated when exceeding max_content_length_per_result."""
    from crewai_tools import ValyuSearchTool

    long_content = "x" * 2000
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com",
                "content": long_content,
            }
        ]
    }
    mock_valyu_class.return_value.search.return_value = mock_response

    tool = ValyuSearchTool(api_key="test_api_key", max_content_length_per_result=100)
    result = tool._run(query="test query")

    parsed_result = json.loads(result)
    content = parsed_result["results"][0]["content"]
    assert len(content) == 103  # 100 chars + "..."
    assert content.endswith("...")


def test_valyu_search_tool_content_not_truncated_when_short(
    mock_valyu_class, mock_valyu_available
):
    """Test that short content is not truncated."""
    from crewai_tools import ValyuSearchTool

    short_content = "Short content"
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [
            {
                "title": "Test Result",
                "url": "https://example.com",
                "content": short_content,
            }
        ]
    }
    mock_valyu_class.return_value.search.return_value = mock_response

    tool = ValyuSearchTool(api_key="test_api_key", max_content_length_per_result=1000)
    result = tool._run(query="test query")

    parsed_result = json.loads(result)
    assert parsed_result["results"][0]["content"] == short_content


def test_valyu_search_tool_run_no_client_raises_error(
    mock_valyu_class, mock_valyu_available
):
    """Test that _run raises error when client is not initialized."""
    from crewai_tools import ValyuSearchTool

    tool = ValyuSearchTool(api_key="test_api_key")
    tool.client = None

    with pytest.raises(ValueError, match="Valyu client is not initialized"):
        tool._run(query="test query")


@pytest.mark.asyncio
async def test_valyu_search_tool_arun(mock_valyu_class, mock_valyu_available):
    """Test the async _arun method."""
    from crewai_tools import ValyuSearchTool

    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "results": [{"title": "Test", "content": "Content"}]
    }
    mock_valyu_class.return_value.search.return_value = mock_response

    tool = ValyuSearchTool(api_key="test_api_key")
    result = await tool._arun(query="test query")

    parsed_result = json.loads(result)
    assert "results" in parsed_result


def test_valyu_search_tool_handles_dict_response(
    mock_valyu_class, mock_valyu_available
):
    """Test that tool handles dict response (legacy .dict() method)."""
    from crewai_tools import ValyuSearchTool

    mock_response = MagicMock(spec=[])
    mock_response.dict = MagicMock(
        return_value={"results": [{"title": "Test", "content": "Content"}]}
    )
    del mock_response.model_dump
    mock_valyu_class.return_value.search.return_value = mock_response

    tool = ValyuSearchTool(api_key="test_api_key")
    result = tool._run(query="test query")

    parsed_result = json.loads(result)
    assert "results" in parsed_result
    mock_response.dict.assert_called_once()


def test_valyu_search_tool_handles_raw_dict_response(
    mock_valyu_class, mock_valyu_available
):
    """Test that tool handles raw dict response."""
    from crewai_tools import ValyuSearchTool

    mock_valyu_class.return_value.search.return_value = {
        "results": [{"title": "Test", "content": "Content"}]
    }

    tool = ValyuSearchTool(api_key="test_api_key")
    result = tool._run(query="test query")

    parsed_result = json.loads(result)
    assert "results" in parsed_result


def test_valyu_search_tool_name_and_description(
    mock_valyu_class, mock_valyu_available
):
    """Test tool has correct name and description."""
    from crewai_tools import ValyuSearchTool

    tool = ValyuSearchTool(api_key="test_api_key")

    assert tool.name == "Valyu Search"
    assert "unified searches" in tool.description
    assert "web" in tool.description
    assert "academic" in tool.description


def test_valyu_search_tool_env_vars_config(mock_valyu_class, mock_valyu_available):
    """Test tool has correct environment variable configuration."""
    from crewai_tools import ValyuSearchTool

    tool = ValyuSearchTool(api_key="test_api_key")

    assert len(tool.env_vars) == 1
    assert tool.env_vars[0].name == "VALYU_API_KEY"
    assert tool.env_vars[0].required is True


def test_valyu_search_tool_package_dependencies(
    mock_valyu_class, mock_valyu_available
):
    """Test tool has correct package dependencies."""
    from crewai_tools import ValyuSearchTool

    tool = ValyuSearchTool(api_key="test_api_key")

    assert "valyu" in tool.package_dependencies
