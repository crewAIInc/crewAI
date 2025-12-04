import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai_tools.tools.airweave_tool.airweave_search_tool import AirweaveSearchTool


@pytest.fixture(autouse=True)
def mock_airweave_api_key():
    """Mock the AIRWEAVE_API_KEY environment variable for all tests."""
    with patch.dict(os.environ, {"AIRWEAVE_API_KEY": "test_key"}):
        yield


@pytest.fixture
def mock_airweave_sdk():
    """Mock the Airweave SDK to avoid import errors."""
    with patch(
        "crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True
    ):
        mock_sdk = MagicMock()
        mock_async_sdk = MagicMock()
        mock_search_request = MagicMock()

        with patch.multiple(
            "crewai_tools.tools.airweave_tool.airweave_search_tool",
            AirweaveSDK=mock_sdk,
            AsyncAirweaveSDK=mock_async_sdk,
            SearchRequest=mock_search_request,
        ):
            yield {
                "sdk": mock_sdk,
                "async_sdk": mock_async_sdk,
                "search_request": mock_search_request,
            }


@pytest.fixture
def airweave_tool(mock_airweave_sdk):
    """Create an AirweaveSearchTool instance for testing."""
    return AirweaveSearchTool()


def test_airweave_tool_initialization_defaults(mock_airweave_sdk):
    """Test tool initialization with default values."""
    tool = AirweaveSearchTool()

    assert tool.name == "Search Airweave Collection"
    assert "Airweave" in tool.description
    assert tool.api_key == "test_key"
    assert tool.base_url is None
    assert tool.framework_name == "crewai"
    assert tool.framework_version is None
    assert tool.timeout == 60
    assert tool.max_content_length_per_result == 1000


def test_airweave_tool_initialization_custom(mock_airweave_sdk):
    """Test tool initialization with custom values."""
    tool = AirweaveSearchTool(
        api_key="custom_key",
        base_url="https://custom.airweave.ai",
        framework_name="custom_framework",
        framework_version="1.0.0",
        timeout=120,
        max_content_length_per_result=2000,
    )

    assert tool.api_key == "custom_key"
    assert tool.base_url == "https://custom.airweave.ai"
    assert tool.framework_name == "custom_framework"
    assert tool.framework_version == "1.0.0"
    assert tool.timeout == 120
    assert tool.max_content_length_per_result == 2000


def test_airweave_tool_schema(mock_airweave_sdk):
    """Test the input schema has required fields."""
    from crewai_tools.tools.airweave_tool.airweave_search_tool import (
        AirweaveSearchToolSchema,
    )

    schema = AirweaveSearchToolSchema

    # Check that required fields are in annotations
    assert "query" in schema.__annotations__
    assert "collection_id" in schema.__annotations__
    assert "limit" in schema.__annotations__
    assert "generate_answer" in schema.__annotations__
    assert "expand_query" in schema.__annotations__
    assert "rerank" in schema.__annotations__
    assert "temporal_relevance" in schema.__annotations__


def test_airweave_tool_search_basic(mock_airweave_sdk, airweave_tool):
    """Test basic search functionality with mocked response."""
    # Mock the search response
    mock_response = MagicMock()
    mock_response.results = [
        {
            "content": "Test document content about Q4 sales",
            "metadata": {"source": "sales-report", "updated_at": "2024-01-01"},
            "score": 0.95,
        },
        {
            "content": "Another test document with sales data",
            "metadata": {"source": "metrics-dashboard", "updated_at": "2024-01-02"},
            "score": 0.87,
        },
    ]
    mock_response.completion = None

    # Mock the collections.search method
    airweave_tool.client = MagicMock()
    airweave_tool.client.collections.search.return_value = mock_response

    # Execute search
    result = airweave_tool._run(
        query="Q4 sales performance", collection_id="sales-data-2024", limit=10
    )

    # Verify the result
    import json

    result_dict = json.loads(result)
    assert result_dict["query"] == "Q4 sales performance"
    assert result_dict["results_count"] == 2
    assert len(result_dict["results"]) == 2
    assert "Q4 sales" in result_dict["results"][0]["content"]
    assert "completion" not in result_dict


def test_airweave_tool_search_with_answer_generation(mock_airweave_sdk, airweave_tool):
    """Test search with AI-generated answer."""
    # Mock the search response with completion
    mock_response = MagicMock()
    mock_response.results = [
        {
            "content": "Test content",
            "metadata": {"source": "test"},
            "score": 0.9,
        }
    ]
    mock_response.completion = (
        "Based on the data, Q4 sales performance exceeded expectations."
    )

    airweave_tool.client = MagicMock()
    airweave_tool.client.collections.search.return_value = mock_response

    # Execute search with answer generation
    result = airweave_tool._run(
        query="Q4 sales summary",
        collection_id="sales-data",
        limit=5,
        generate_answer=True,
    )

    # Verify the completion is included
    import json

    result_dict = json.loads(result)
    assert "completion" in result_dict
    assert "exceeded expectations" in result_dict["completion"]


def test_airweave_tool_content_truncation(mock_airweave_sdk, airweave_tool):
    """Test that long content is truncated properly."""
    # Create a very long content string
    long_content = "A" * 2000

    mock_response = MagicMock()
    mock_response.results = [
        {
            "content": long_content,
            "metadata": {"source": "test"},
            "score": 0.9,
        }
    ]
    mock_response.completion = None

    airweave_tool.client = MagicMock()
    airweave_tool.client.collections.search.return_value = mock_response
    airweave_tool.max_content_length_per_result = 1000

    # Execute search
    result = airweave_tool._run(query="test", collection_id="test-collection", limit=5)

    # Verify content was truncated
    import json

    result_dict = json.loads(result)
    result_content = result_dict["results"][0]["content"]
    assert len(result_content) <= 1003  # 1000 + "..."
    assert result_content.endswith("...")


def test_airweave_tool_search_with_all_parameters(mock_airweave_sdk, airweave_tool):
    """Test search with all optional parameters."""
    mock_response = MagicMock()
    mock_response.results = [{"content": "Test", "metadata": {}, "score": 0.9}]
    mock_response.completion = None

    airweave_tool.client = MagicMock()
    airweave_tool.client.collections.search.return_value = mock_response

    # Execute search with all parameters
    result = airweave_tool._run(
        query="comprehensive search",
        collection_id="test-collection",
        limit=15,
        generate_answer=True,
        expand_query=True,
        rerank=True,
        temporal_relevance=0.5,
    )

    # Verify the call was made
    assert airweave_tool.client.collections.search.called
    import json

    result_dict = json.loads(result)
    assert result_dict["query"] == "comprehensive search"


def test_airweave_tool_error_handling(mock_airweave_sdk, airweave_tool):
    """Test error handling when API call fails."""
    # Mock an API error
    airweave_tool.client = MagicMock()
    airweave_tool.client.collections.search.side_effect = Exception(
        "Collection not found"
    )

    # Execute search (should not raise, should return error JSON)
    result = airweave_tool._run(
        query="test", collection_id="nonexistent-collection", limit=10
    )

    # Verify error is returned in JSON format
    import json

    result_dict = json.loads(result)
    assert "error" in result_dict
    assert "Collection not found" in result_dict["error"]
    assert result_dict["query"] == "test"
    assert result_dict["collection_id"] == "nonexistent-collection"


def test_airweave_tool_client_not_initialized(mock_airweave_sdk):
    """Test error handling when client is not initialized."""
    tool = AirweaveSearchTool()
    tool.client = None  # Simulate uninitialized client

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        tool._run(query="test", collection_id="test", limit=10)

    assert "not initialized" in str(exc_info.value)


@pytest.mark.asyncio
async def test_airweave_tool_async_search(mock_airweave_sdk, airweave_tool):
    """Test asynchronous search functionality."""
    # Mock the async search response
    mock_response = MagicMock()
    mock_response.results = [
        {"content": "Async test content", "metadata": {}, "score": 0.9}
    ]
    mock_response.completion = None

    # Mock the async client
    airweave_tool.async_client = MagicMock()
    airweave_tool.async_client.collections.search = AsyncMock(
        return_value=mock_response
    )

    # Execute async search
    result = await airweave_tool._arun(
        query="async test", collection_id="test-collection", limit=10
    )

    # Verify the result
    import json

    result_dict = json.loads(result)
    assert result_dict["query"] == "async test"
    assert result_dict["results_count"] == 1
    assert "Async test content" in result_dict["results"][0]["content"]


@pytest.mark.asyncio
async def test_airweave_tool_async_error_handling(mock_airweave_sdk, airweave_tool):
    """Test error handling in async search."""
    # Mock an async API error
    airweave_tool.async_client = MagicMock()
    airweave_tool.async_client.collections.search = AsyncMock(
        side_effect=Exception("Async error")
    )

    # Execute async search (should not raise, should return error JSON)
    result = await airweave_tool._arun(
        query="test", collection_id="test-collection", limit=10
    )

    # Verify error is returned in JSON format
    import json

    result_dict = json.loads(result)
    assert "error" in result_dict
    assert "Async error" in result_dict["error"]


@pytest.mark.asyncio
async def test_airweave_tool_async_client_not_initialized(mock_airweave_sdk):
    """Test error handling when async client is not initialized."""
    tool = AirweaveSearchTool()
    tool.async_client = None  # Simulate uninitialized async client

    # Should raise ValueError
    with pytest.raises(ValueError) as exc_info:
        await tool._arun(query="test", collection_id="test", limit=10)

    assert "not initialized" in str(exc_info.value)


def test_airweave_tool_format_results(mock_airweave_sdk, airweave_tool):
    """Test the _format_results method."""
    mock_response = MagicMock()
    mock_response.results = [
        {"content": "Test 1", "score": 0.9},
        {"content": "Test 2", "score": 0.8},
    ]
    mock_response.completion = "This is a test completion"

    result_str = airweave_tool._format_results(mock_response, "test query")

    import json

    result_dict = json.loads(result_str)

    assert result_dict["query"] == "test query"
    assert result_dict["results_count"] == 2
    assert result_dict["completion"] == "This is a test completion"
    assert len(result_dict["results"]) == 2


def test_airweave_tool_env_vars(mock_airweave_sdk, airweave_tool):
    """Test that env_vars are properly configured."""
    assert len(airweave_tool.env_vars) == 1
    assert airweave_tool.env_vars[0].name == "AIRWEAVE_API_KEY"
    assert airweave_tool.env_vars[0].required is True


def test_airweave_tool_package_dependencies(mock_airweave_sdk, airweave_tool):
    """Test that package dependencies are specified."""
    assert "airweave-sdk" in airweave_tool.package_dependencies


def test_airweave_tool_empty_results(mock_airweave_sdk, airweave_tool):
    """Test handling of empty search results."""
    mock_response = MagicMock()
    mock_response.results = []
    mock_response.completion = None

    airweave_tool.client = MagicMock()
    airweave_tool.client.collections.search.return_value = mock_response

    result = airweave_tool._run(query="test", collection_id="test-collection", limit=10)

    import json

    result_dict = json.loads(result)
    assert result_dict["results_count"] == 0
    assert result_dict["results"] == []


def test_airweave_tool_with_filters(mock_airweave_sdk, airweave_tool):
    """Test search with temporal relevance parameter."""
    mock_response = MagicMock()
    mock_response.results = [{"content": "Recent content", "score": 0.95}]
    mock_response.completion = None

    airweave_tool.client = MagicMock()
    airweave_tool.client.collections.search.return_value = mock_response

    # Execute search with temporal relevance
    result = airweave_tool._run(
        query="recent news",
        collection_id="news-2024",
        limit=10,
        temporal_relevance=0.7,  # Favor recent content
    )

    import json

    result_dict = json.loads(result)
    assert result_dict["query"] == "recent news"
    assert result_dict["results_count"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

