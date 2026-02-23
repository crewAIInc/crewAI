"""Tests for NimbleSearchTool."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_search_response():
  """Create a mock search response matching actual Nimble API structure."""
  response_data = {
    "results": [
      {
        "title": "Test Result 1",
        "url": "https://example.com/1",
        "content": "This is test content for result 1",
        "description": "Test description 1",
        "metadata": {"country": "US", "locale": "en", "position": 1},
        "extra_fields": None,
      },
      {
        "title": "Test Result 2",
        "url": "https://example.com/2",
        "content": "This is test content for result 2",
        "description": "Test description 2",
        "metadata": {"country": "US", "locale": "en", "position": 2},
        "extra_fields": None,
      },
    ],
    "query": "test query",
  }
  # Mock SearchResponse object with model_dump method
  mock_response = MagicMock()
  mock_response.model_dump.return_value = response_data
  return mock_response


@pytest.fixture
def mock_search_response_with_long_content():
  """Create a mock search response with long content."""
  response_data = {
    "results": [
      {
        "title": "Long Content Result",
        "url": "https://example.com/long",
        "content": "A" * 2000,  # Content longer than default max
        "description": "Long content description",
        "metadata": {"country": "US", "locale": "en", "position": 1},
        "extra_fields": None,
      },
    ],
    "query": "test query",
  }
  # Mock SearchResponse object with model_dump method
  mock_response = MagicMock()
  mock_response.model_dump.return_value = response_data
  return mock_response


@patch("nimble_python.AsyncNimble")
@patch("nimble_python.Nimble")
def test_nimble_search_tool_initialization(mock_nimble, mock_async_nimble):
  """Test that NimbleSearchTool initializes with correct default values."""
  from crewai_tools.tools.nimble_search_tool.nimble_search_tool import (
    NimbleSearchTool,
  )

  tool = NimbleSearchTool(api_key="test-key")

  # Check default values
  assert tool.name == "Nimble Search"
  assert tool.max_results == 3
  assert tool.deep_search is False
  assert tool.include_answer is False
  assert tool.parsing_type == "markdown"
  assert tool.max_content_length_per_result == 1000

  # Verify clients were initialized with tracking headers
  mock_nimble.assert_called_once()
  mock_async_nimble.assert_called_once()

  # Get the call arguments
  nimble_call_kwargs = mock_nimble.call_args[1]
  assert nimble_call_kwargs["api_key"] == "test-key"
  assert "default_headers" in nimble_call_kwargs
  assert nimble_call_kwargs["default_headers"]["X-Client-Source"] == "crewai-tools"
  assert (
    nimble_call_kwargs["default_headers"]["X-Client-Tool"] == "NimbleSearchTool"
  )


def test_nimble_search_tool_schema():
  """Test that the input schema is correctly defined."""
  from crewai_tools.tools.nimble_search_tool.nimble_search_tool import (
    NimbleSearchToolSchema,
  )

  # Test schema structure
  schema = NimbleSearchToolSchema(query="test query")
  assert schema.query == "test query"

  # Test that query is required
  with pytest.raises(Exception):  # Pydantic validation error
    NimbleSearchToolSchema()


@patch("nimble_python.AsyncNimble")
@patch("nimble_python.Nimble")
def test_nimble_search_tool_run(
  mock_nimble_class, mock_async_nimble_class, mock_search_response
):
  """Test synchronous search execution."""
  from crewai_tools.tools.nimble_search_tool.nimble_search_tool import (
    NimbleSearchTool,
  )

  # Setup mock
  mock_client = MagicMock()
  mock_client.search.return_value = mock_search_response
  mock_nimble_class.return_value = mock_client

  # Create tool and execute search
  tool = NimbleSearchTool(api_key="test-key", max_results=2)
  result = tool._run(query="test query")

  # Verify search was called with correct parameters
  mock_client.search.assert_called_once_with(
    query="test query",
    num_results=2,
    deep_search=False,
    include_answer=False,
    parsing_type="markdown",
  )

  # Verify result is valid JSON
  parsed_result = json.loads(result)
  assert "results" in parsed_result
  assert len(parsed_result["results"]) == 2
  assert parsed_result["results"][0]["title"] == "Test Result 1"


@patch("nimble_python.AsyncNimble")
@patch("nimble_python.Nimble")
@pytest.mark.asyncio
async def test_nimble_search_tool_arun(
  mock_nimble_class, mock_async_nimble_class, mock_search_response
):
  """Test asynchronous search execution."""
  from crewai_tools.tools.nimble_search_tool.nimble_search_tool import (
    NimbleSearchTool,
  )

  # Setup mock
  mock_async_client = AsyncMock()
  mock_async_client.search.return_value = mock_search_response
  mock_async_nimble_class.return_value = mock_async_client

  # Create tool and execute async search
  tool = NimbleSearchTool(api_key="test-key", max_results=2)
  result = await tool._arun(query="test query")

  # Verify search was called with correct parameters
  mock_async_client.search.assert_called_once_with(
    query="test query",
    num_results=2,
    deep_search=False,
    include_answer=False,
    parsing_type="markdown",
  )

  # Verify result is valid JSON
  parsed_result = json.loads(result)
  assert "results" in parsed_result
  assert len(parsed_result["results"]) == 2


@patch("nimble_python.AsyncNimble")
@patch("nimble_python.Nimble")
def test_nimble_search_tool_content_truncation(
  mock_nimble_class,
  mock_async_nimble_class,
  mock_search_response_with_long_content,
):
  """Test that content is truncated to max_content_length_per_result."""
  from crewai_tools.tools.nimble_search_tool.nimble_search_tool import (
    NimbleSearchTool,
  )

  # Setup mock
  mock_client = MagicMock()
  mock_client.search.return_value = mock_search_response_with_long_content
  mock_nimble_class.return_value = mock_client

  # Create tool with specific max content length
  tool = NimbleSearchTool(api_key="test-key", max_content_length_per_result=500)
  result = tool._run(query="test query")

  # Verify content was truncated
  parsed_result = json.loads(result)
  content = parsed_result["results"][0]["content"]

  # Content should be truncated to 500 + "..." (503 total)
  assert len(content) == 503
  assert content.endswith("...")
  assert content.startswith("A" * 500)


@patch("nimble_python.AsyncNimble")
@patch("nimble_python.Nimble")
def test_nimble_search_tool_with_filters(
  mock_nimble_class, mock_async_nimble_class, mock_search_response
):
  """Test search with various filter options."""
  from crewai_tools.tools.nimble_search_tool.nimble_search_tool import (
    NimbleSearchTool,
  )

  # Setup mock
  mock_client = MagicMock()
  mock_client.search.return_value = mock_search_response
  mock_nimble_class.return_value = mock_client

  # Create tool with config defaults
  tool = NimbleSearchTool(
    api_key="test-key",
    max_results=5,
    deep_search=True,
    include_answer=True,
    parsing_type="plain_text",
    locale="en-US",
    country="US",
  )

  # Pass runtime filters when calling
  result = tool._run(
    query="filtered query",
    time_range="week",
    include_domains=["example.com", "test.com"],
    exclude_domains=["spam.com"],
  )

  # Verify search was called with all filters
  mock_client.search.assert_called_once_with(
    query="filtered query",
    num_results=5,
    deep_search=True,
    include_answer=True,
    time_range="week",
    include_domains=["example.com", "test.com"],
    exclude_domains=["spam.com"],
    parsing_type="plain_text",
    locale="en-US",
    country="US",
  )

  # Verify result is valid
  parsed_result = json.loads(result)
  assert "results" in parsed_result


@patch("nimble_python.AsyncNimble")
@patch("nimble_python.Nimble")
def test_nimble_search_tool_missing_client(mock_nimble_class, mock_async_nimble_class):
  """Test error handling when client is not initialized."""
  from crewai_tools.tools.nimble_search_tool.nimble_search_tool import (
    NimbleSearchTool,
  )

  # Create tool but manually set client to None
  tool = NimbleSearchTool(api_key="test-key")
  tool.client = None

  # Verify error is raised
  with pytest.raises(ValueError) as exc_info:
    tool._run(query="test query")

  assert "Nimble client is not initialized" in str(exc_info.value)


@patch("nimble_python.AsyncNimble")
@patch("nimble_python.Nimble")
@pytest.mark.asyncio
async def test_nimble_search_tool_missing_async_client(
  mock_nimble_class, mock_async_nimble_class
):
  """Test error handling when async client is not initialized."""
  from crewai_tools.tools.nimble_search_tool.nimble_search_tool import (
    NimbleSearchTool,
  )

  # Create tool but manually set async client to None
  tool = NimbleSearchTool(api_key="test-key")
  tool.async_client = None

  # Verify error is raised
  with pytest.raises(ValueError) as exc_info:
    await tool._arun(query="test query")

  assert "Nimble async client is not initialized" in str(exc_info.value)


def test_nimble_search_tool_missing_package():
  """Test error handling when nimble-python package is not installed."""
  import builtins

  original_import = builtins.__import__

  def mock_import(name, *args, **kwargs):
    if name == "nimble_python":
      raise ImportError("No module named 'nimble_python'")
    return original_import(name, *args, **kwargs)

  with patch("builtins.__import__", side_effect=mock_import), patch(
    "click.confirm", return_value=False
  ):
    from crewai_tools.tools.nimble_search_tool.nimble_search_tool import (
      NimbleSearchTool,
    )

    with pytest.raises(ImportError) as exc_info:
      NimbleSearchTool(api_key="test-key")

    assert "nimble-python" in str(exc_info.value).lower()
