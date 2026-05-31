"""Tests for Scavio tools."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai_tools.tools.scavio_tool.scavio_amazon_search_tool import (
    ScavioAmazonSearchTool,
)
from crewai_tools.tools.scavio_tool.scavio_search_tool import ScavioSearchTool
from crewai_tools.tools.scavio_tool.scavio_youtube_search_tool import (
    ScavioYouTubeSearchTool,
)

MOCK_API_KEY = "sk_live_test_key_12345"


def _mock_search_response():
    return {
        "query": "test query",
        "page": 1,
        "credits_used": 1,
        "credits_remaining": 999,
        "results": [
            {
                "title": f"Result {i}",
                "url": f"https://example.com/{i}",
                "description": f"Description {i}",
                "position": i,
            }
            for i in range(1, 11)
        ],
        "knowledge_graph": {
            "title": "Test Subject",
            "subtitle": "A test entity",
        },
        "questions": [
            {"question": "What is test?", "answer": "A test."},
        ],
    }


def _mock_amazon_response():
    return {
        "credits_used": 1,
        "data": {
            "products": [
                {
                    "asin": f"B00{i}",
                    "title": f"Product {i}",
                    "price": {"value": 29.99 + i, "currency": "USD"},
                    "rating": 4.5,
                    "reviews_count": 100 + i,
                }
                for i in range(1, 11)
            ],
        },
    }


def _mock_youtube_response():
    return {
        "credits_used": 1,
        "data": [
            {
                "video_id": f"vid_{i}",
                "title": f"Video {i}",
                "channel": f"Channel {i}",
                "view_count": 1000 * i,
            }
            for i in range(1, 11)
        ],
    }


class TestScavioSearchTool:
    """Tests for ScavioSearchTool."""

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_initialization(self, mock_async_client, mock_client):
        """Test default initialization values."""
        tool = ScavioSearchTool(api_key=MOCK_API_KEY)
        assert tool.name == "Scavio Web Search"
        assert tool.max_results == 5
        assert tool.search_type == "classic"
        assert tool.include_knowledge_graph is True
        assert tool.include_questions is True

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_custom_params(self, mock_async_client, mock_client):
        """Test custom parameter assignment."""
        tool = ScavioSearchTool(
            api_key=MOCK_API_KEY,
            max_results=10,
            search_type="news",
            country_code="us",
            language="en",
            include_knowledge_graph=False,
        )
        assert tool.max_results == 10
        assert tool.search_type == "news"
        assert tool.country_code == "us"
        assert tool.language == "en"
        assert tool.include_knowledge_graph is False

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_run_truncates_results(self, mock_async_client, mock_client_cls):
        """Test that results are truncated to max_results."""
        mock_client = MagicMock()
        mock_client.google.search.return_value = _mock_search_response()
        mock_client_cls.return_value = mock_client

        tool = ScavioSearchTool(api_key=MOCK_API_KEY, max_results=3)
        result = tool._run(query="test query")
        parsed = json.loads(result)

        assert len(parsed["results"]) == 3
        mock_client.google.search.assert_called_once_with(
            query="test query",
            search_type="classic",
            country_code=None,
            language=None,
            device="desktop",
        )

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_run_strips_knowledge_graph(self, mock_async_client, mock_client_cls):
        """Test that knowledge graph and questions can be excluded."""
        mock_client = MagicMock()
        mock_client.google.search.return_value = _mock_search_response()
        mock_client_cls.return_value = mock_client

        tool = ScavioSearchTool(
            api_key=MOCK_API_KEY,
            include_knowledge_graph=False,
            include_questions=False,
        )
        result = tool._run(query="test query")
        parsed = json.loads(result)

        assert "knowledge_graph" not in parsed
        assert "questions" not in parsed

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_run_returns_json_string(self, mock_async_client, mock_client_cls):
        """Test that output is a valid JSON string."""
        mock_client = MagicMock()
        mock_client.google.search.return_value = _mock_search_response()
        mock_client_cls.return_value = mock_client

        tool = ScavioSearchTool(api_key=MOCK_API_KEY)
        result = tool._run(query="test query")

        assert isinstance(result, str)
        parsed = json.loads(result)
        assert "results" in parsed

    @pytest.mark.asyncio
    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    async def test_arun_truncates_results(self, mock_async_client_cls, mock_client):
        """Test async path truncates results."""
        mock_async_client = MagicMock()
        mock_async_client.google.search = AsyncMock(
            return_value=_mock_search_response()
        )
        mock_async_client_cls.return_value = mock_async_client

        tool = ScavioSearchTool(api_key=MOCK_API_KEY, max_results=2)
        result = await tool._arun(query="test query")
        parsed = json.loads(result)

        assert len(parsed["results"]) == 2


class TestScavioAmazonSearchTool:
    """Tests for ScavioAmazonSearchTool."""

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_initialization(self, mock_async_client, mock_client):
        """Test default initialization values."""
        tool = ScavioAmazonSearchTool(api_key=MOCK_API_KEY)
        assert tool.name == "Scavio Amazon Search"
        assert tool.domain == "com"
        assert tool.max_results == 5

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_run_truncates_products(self, mock_async_client, mock_client_cls):
        """Test that products are truncated to max_results."""
        mock_client = MagicMock()
        mock_client.amazon.search.return_value = _mock_amazon_response()
        mock_client_cls.return_value = mock_client

        tool = ScavioAmazonSearchTool(api_key=MOCK_API_KEY, max_results=3)
        result = tool._run(query="headphones")
        parsed = json.loads(result)

        assert len(parsed["data"]["products"]) == 3
        mock_client.amazon.search.assert_called_once_with(
            query="headphones",
            domain="com",
            sort_by=None,
        )

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_custom_domain(self, mock_async_client, mock_client_cls):
        """Test custom Amazon domain parameter."""
        mock_client = MagicMock()
        mock_client.amazon.search.return_value = _mock_amazon_response()
        mock_client_cls.return_value = mock_client

        tool = ScavioAmazonSearchTool(api_key=MOCK_API_KEY, domain="co.uk")
        tool._run(query="tea")

        mock_client.amazon.search.assert_called_once_with(
            query="tea",
            domain="co.uk",
            sort_by=None,
        )

    @pytest.mark.asyncio
    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    async def test_arun_truncates_products(self, mock_async_client_cls, mock_client):
        """Test async path truncates products."""
        mock_async_client = MagicMock()
        mock_async_client.amazon.search = AsyncMock(
            return_value=_mock_amazon_response()
        )
        mock_async_client_cls.return_value = mock_async_client

        tool = ScavioAmazonSearchTool(api_key=MOCK_API_KEY, max_results=2)
        result = await tool._arun(query="headphones")
        parsed = json.loads(result)

        assert len(parsed["data"]["products"]) == 2


class TestScavioYouTubeSearchTool:
    """Tests for ScavioYouTubeSearchTool."""

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_initialization(self, mock_async_client, mock_client):
        """Test default initialization values."""
        tool = ScavioYouTubeSearchTool(api_key=MOCK_API_KEY)
        assert tool.name == "Scavio YouTube Search"
        assert tool.max_results == 5

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_run_truncates_data(self, mock_async_client, mock_client_cls):
        """Test that data is truncated to max_results."""
        mock_client = MagicMock()
        mock_client.youtube.search.return_value = _mock_youtube_response()
        mock_client_cls.return_value = mock_client

        tool = ScavioYouTubeSearchTool(api_key=MOCK_API_KEY, max_results=3)
        result = tool._run(query="python tutorial")
        parsed = json.loads(result)

        assert len(parsed["data"]) == 3

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_run_with_filters(self, mock_async_client, mock_client_cls):
        """Test filter parameters are passed to API."""
        mock_client = MagicMock()
        mock_client.youtube.search.return_value = _mock_youtube_response()
        mock_client_cls.return_value = mock_client

        tool = ScavioYouTubeSearchTool(
            api_key=MOCK_API_KEY,
            upload_date="week",
            sort_by="views",
            type="video",
            duration="medium",
        )
        tool._run(query="crewai tutorial")

        mock_client.youtube.search.assert_called_once_with(
            query="crewai tutorial",
            upload_date="week",
            sort_by="views",
            type="video",
            duration="medium",
        )

    @pytest.mark.asyncio
    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    async def test_arun_truncates_data(self, mock_async_client_cls, mock_client):
        """Test async path truncates data."""
        mock_async_client = MagicMock()
        mock_async_client.youtube.search = AsyncMock(
            return_value=_mock_youtube_response()
        )
        mock_async_client_cls.return_value = mock_async_client

        tool = ScavioYouTubeSearchTool(api_key=MOCK_API_KEY, max_results=2)
        result = await tool._arun(query="python tutorial")
        parsed = json.loads(result)

        assert len(parsed["data"]) == 2


class TestScavioToolEnvVar:
    """Tests for API key resolution from environment."""

    @patch.dict("os.environ", {"SCAVIO_API_KEY": MOCK_API_KEY})
    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_api_key_from_env(self, mock_async_client, mock_client):
        """Test that api_key is resolved from SCAVIO_API_KEY env var."""
        tool = ScavioSearchTool()
        assert tool.api_key == MOCK_API_KEY


class TestScavioToolValidation:
    """Tests for initialization validation."""

    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", False)
    def test_raises_import_error_when_package_missing(self):
        """Test that ImportError is raised when scavio is not installed."""
        with pytest.raises(ImportError, match="scavio"):
            ScavioSearchTool(api_key=MOCK_API_KEY)

    @patch("crewai_tools.tools.scavio_tool.base.ScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.AsyncScavioClient")
    @patch("crewai_tools.tools.scavio_tool.base.SCAVIO_AVAILABLE", True)
    def test_raises_value_error_when_api_key_missing(
        self, mock_async_client, mock_client
    ):
        """Test that ValueError is raised when api_key is not provided."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key"):
                ScavioSearchTool(api_key=None)

    def test_rejects_negative_max_results(self):
        """Test that max_results rejects negative values."""
        with pytest.raises(Exception):
            ScavioSearchTool(api_key=MOCK_API_KEY, max_results=-1)

    def test_rejects_zero_max_results(self):
        """Test that max_results rejects zero."""
        with pytest.raises(Exception):
            ScavioSearchTool(api_key=MOCK_API_KEY, max_results=0)
