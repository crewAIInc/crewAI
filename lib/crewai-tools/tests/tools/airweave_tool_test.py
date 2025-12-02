"""Unit tests for Airweave tools."""

import os
from unittest.mock import Mock, patch

import pytest

from crewai_tools.tools.airweave_tool import (
    AirweaveAdvancedSearchTool,
    AirweaveSearchTool,
)


@pytest.fixture
def mock_env(monkeypatch):
    """Set up environment variables."""
    monkeypatch.setenv("AIRWEAVE_API_KEY", "test_api_key_12345")


@pytest.fixture
def mock_search_response():
    """Create mock search response with raw results."""
    return Mock(
        status="success",
        results=[
            {
                "score": 0.95,
                "payload": {
                    "md_content": "Test content from Stripe about a customer payment",
                    "source_name": "Stripe",
                    "entity_id": "cus_123",
                    "created_at": "2024-01-15T10:00:00Z",
                    "url": "https://stripe.com/customers/cus_123"
                }
            },
            {
                "score": 0.87,
                "payload": {
                    "md_content": "GitHub issue about payment integration bug",
                    "source_name": "GitHub",
                    "entity_id": "issue_456",
                    "created_at": "2024-01-14T15:30:00Z"
                }
            }
        ],
        response_type="raw",
        completion=None
    )


@pytest.fixture
def mock_completion_response():
    """Create mock search response with completion."""
    return Mock(
        status="success",
        results=[],
        response_type="completion",
        completion="Based on the data from Stripe and GitHub, there were 3 failed payments in the last month due to expired cards."
    )


@pytest.fixture
def mock_no_results_response():
    """Create mock response with no results."""
    return Mock(
        status="no_results",
        results=[],
        response_type="raw",
        completion=None
    )


class TestAirweaveSearchTool:
    """Tests for AirweaveSearchTool."""

    def test_requires_api_key(self, monkeypatch):
        """Test that tool requires API key."""
        monkeypatch.delenv("AIRWEAVE_API_KEY", raising=False)
        with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AirweaveSDK"):
                with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AsyncAirweaveSDK"):
                    with pytest.raises(ValueError, match="AIRWEAVE_API_KEY"):
                        AirweaveSearchTool(collection_id="test-collection")

    def test_initialization_with_valid_api_key(self, mock_env):
        """Test successful initialization with API key."""
        with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AirweaveSDK"):
                with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AsyncAirweaveSDK"):
                    tool = AirweaveSearchTool(collection_id="test-collection")
                    assert tool.collection_id == "test-collection"
                    assert tool.name == "Airweave Search"

    def test_basic_search_raw_results(self, mock_env, mock_search_response):
        """Test basic search with raw results."""
        with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search.return_value = mock_search_response
                    MockSDK.return_value = mock_client

                    tool = AirweaveSearchTool(collection_id="test-collection")
                    result = tool.run(query="find failed payments", limit=5)

                    # Verify API call
                    mock_client.collections.search.assert_called_once_with(
                        readable_id="test-collection",
                        query="find failed payments",
                        limit=5,
                        offset=0,
                        response_type="raw",
                        recency_bias=0.0
                    )

                    # Verify result format
                    assert "Found 2 result" in result
                    assert "Test content from Stripe" in result
                    assert "Stripe" in result
                    assert "0.950" in result
                    assert "GitHub issue" in result

    def test_search_with_completion(self, mock_env, mock_completion_response):
        """Test search requesting AI-generated completion."""
        with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search.return_value = mock_completion_response
                    MockSDK.return_value = mock_client

                    tool = AirweaveSearchTool(collection_id="test-collection")
                    result = tool.run(query="what are the payment issues?", response_type="completion")

                    # Verify API call
                    mock_client.collections.search.assert_called_once_with(
                        readable_id="test-collection",
                        query="what are the payment issues?",
                        limit=10,
                        offset=0,
                        response_type="completion",
                        recency_bias=0.0
                    )

                    # Verify completion response
                    assert "Based on the data from Stripe and GitHub" in result
                    assert "3 failed payments" in result

    def test_no_results_handling(self, mock_env, mock_no_results_response):
        """Test handling of no results."""
        with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search.return_value = mock_no_results_response
                    MockSDK.return_value = mock_client

                    tool = AirweaveSearchTool(collection_id="test-collection")
                    result = tool.run(query="nonexistent query")

                    assert "No results found" in result

    def test_no_relevant_results_handling(self, mock_env):
        """Test handling of no relevant results."""
        mock_response = Mock(
            status="no_relevant_results",
            results=[],
            response_type="raw",
            completion=None
        )

        with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search.return_value = mock_response
                    MockSDK.return_value = mock_client

                    tool = AirweaveSearchTool(collection_id="test-collection")
                    result = tool.run(query="vague query")

                    assert "no sufficiently relevant results" in result

    def test_error_handling(self, mock_env):
        """Test API error handling."""
        with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search.side_effect = Exception("API Error: Collection not found")
                    MockSDK.return_value = mock_client

                    tool = AirweaveSearchTool(collection_id="test-collection")
                    result = tool.run(query="test query")

                    assert "Error performing search" in result
                    assert "API Error" in result

    def test_custom_base_url(self, mock_env):
        """Test initialization with custom base URL."""
        with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AsyncAirweaveSDK"):
                    tool = AirweaveSearchTool(
                        collection_id="test-collection",
                        base_url="http://localhost:8001"
                    )

                    # Verify SDK initialized with custom URL
                    MockSDK.assert_called_once()
                    call_kwargs = MockSDK.call_args[1]
                    assert call_kwargs["base_url"] == "http://localhost:8001"

    def test_recency_bias(self, mock_env, mock_search_response):
        """Test search with recency bias."""
        with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search.return_value = mock_search_response
                    MockSDK.return_value = mock_client

                    tool = AirweaveSearchTool(collection_id="test-collection")
                    tool.run(query="recent issues", recency_bias=0.8)

                    # Verify recency_bias passed correctly
                    call_kwargs = mock_client.collections.search.call_args[1]
                    assert call_kwargs["recency_bias"] == 0.8


class TestAirweaveAdvancedSearchTool:
    """Tests for AirweaveAdvancedSearchTool."""

    def test_requires_api_key(self, monkeypatch):
        """Test that tool requires API key."""
        monkeypatch.delenv("AIRWEAVE_API_KEY", raising=False)
        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AirweaveSDK"):
                with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AsyncAirweaveSDK"):
                    with pytest.raises(ValueError, match="AIRWEAVE_API_KEY"):
                        AirweaveAdvancedSearchTool(collection_id="test-collection")

    def test_initialization(self, mock_env):
        """Test successful initialization."""
        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AirweaveSDK"):
                with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AsyncAirweaveSDK"):
                    tool = AirweaveAdvancedSearchTool(collection_id="test-collection")
                    assert tool.collection_id == "test-collection"
                    assert tool.name == "Airweave Advanced Search"

    def test_advanced_search_with_source_filter(self, mock_env, mock_search_response):
        """Test advanced search with source filtering."""
        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AsyncAirweaveSDK"):
                    with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.Filter") as MockFilter:
                        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.FieldCondition") as MockFieldCondition:
                            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.MatchValue") as MockMatchValue:
                                mock_client = Mock()
                                mock_client.collections.search_advanced.return_value = mock_search_response
                                MockSDK.return_value = mock_client

                                tool = AirweaveAdvancedSearchTool(collection_id="test-collection")
                                result = tool.run(
                                    query="payment issues",
                                    source_filter="Stripe",
                                    limit=10,
                                    enable_reranking=True
                                )

                                # Verify API call with filter
                                mock_client.collections.search_advanced.assert_called_once()
                                call_kwargs = mock_client.collections.search_advanced.call_args[1]
                                assert call_kwargs["query"] == "payment issues"
                                assert call_kwargs["limit"] == 10
                                assert call_kwargs["enable_reranking"] is True
                                assert call_kwargs["filter"] is not None  # Filter object created

                                # Verify result includes source info
                                assert "from Stripe" in result

    def test_advanced_search_with_score_threshold(self, mock_env, mock_search_response):
        """Test advanced search with score threshold."""
        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search_advanced.return_value = mock_search_response
                    MockSDK.return_value = mock_client

                    tool = AirweaveAdvancedSearchTool(collection_id="test-collection")
                    tool.run(query="test", score_threshold=0.8)

                    call_kwargs = mock_client.collections.search_advanced.call_args[1]
                    assert call_kwargs["score_threshold"] == 0.8

    def test_advanced_search_with_search_method(self, mock_env, mock_search_response):
        """Test advanced search with different search methods."""
        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search_advanced.return_value = mock_search_response
                    MockSDK.return_value = mock_client

                    tool = AirweaveAdvancedSearchTool(collection_id="test-collection")

                    # Test neural search
                    tool.run(query="test", search_method="neural")
                    call_kwargs = mock_client.collections.search_advanced.call_args[1]
                    assert call_kwargs["search_method"] == "neural"

    def test_advanced_search_completion_mode(self, mock_env, mock_completion_response):
        """Test advanced search with completion response type."""
        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AsyncAirweaveSDK"):
                    with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.Filter") as MockFilter:
                        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.FieldCondition") as MockFieldCondition:
                            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.MatchValue") as MockMatchValue:
                                mock_client = Mock()
                                mock_client.collections.search_advanced.return_value = mock_completion_response
                                MockSDK.return_value = mock_client

                                tool = AirweaveAdvancedSearchTool(collection_id="test-collection")
                                result = tool.run(
                                    query="summarize payment issues",
                                    response_type="completion",
                                    source_filter="Stripe"
                                )

                                assert "Based on the data" in result
                                assert "3 failed payments" in result

    def test_advanced_search_no_results(self, mock_env, mock_no_results_response):
        """Test advanced search with no results."""
        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search_advanced.return_value = mock_no_results_response
                    MockSDK.return_value = mock_client

                    tool = AirweaveAdvancedSearchTool(collection_id="test-collection")
                    result = tool.run(query="nonexistent", score_threshold=0.99)

                    assert "No results found" in result

    def test_advanced_search_error_handling(self, mock_env):
        """Test advanced search error handling."""
        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search_advanced.side_effect = Exception("Filter error")
                    MockSDK.return_value = mock_client

                    tool = AirweaveAdvancedSearchTool(collection_id="test-collection")
                    result = tool.run(query="test")

                    assert "Error performing advanced search" in result

    def test_recency_bias_default(self, mock_env, mock_search_response):
        """Test that advanced search has default recency bias of 0.3."""
        with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AIRWEAVE_AVAILABLE", True):
            with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AirweaveSDK") as MockSDK:
                with patch("crewai_tools.tools.airweave_tool.airweave_advanced_search_tool.AsyncAirweaveSDK"):
                    mock_client = Mock()
                    mock_client.collections.search_advanced.return_value = mock_search_response
                    MockSDK.return_value = mock_client

                    tool = AirweaveAdvancedSearchTool(collection_id="test-collection")
                    tool.run(query="test")

                    call_kwargs = mock_client.collections.search_advanced.call_args[1]
                    assert call_kwargs["recency_bias"] == 0.3
