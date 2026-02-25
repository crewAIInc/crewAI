"""Tests for OlostepTool and related tools.

This module contains comprehensive tests for the Olostep integration with CrewAI,
covering all modes: scrape, crawl, map, and answer.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from crewai_tools import (
    OlostepAnswerTool,
    OlostepCrawlTool,
    OlostepMapTool,
    OlostepScrapeTool,
    OlostepTool,
)


class TestOlostepToolSchema:
    """Tests for OlostepToolSchema validation."""

    def test_schema_default_mode_is_scrape(self):
        """Test that default mode is scrape."""
        from crewai_tools.tools.olostep_tool.olostep_tool import OlostepToolSchema

        schema = OlostepToolSchema(url="https://example.com")
        assert schema.mode == "scrape"

    def test_schema_accepts_all_modes(self):
        """Test that schema accepts all valid modes."""
        from crewai_tools.tools.olostep_tool.olostep_tool import OlostepToolSchema

        for mode in ["scrape", "crawl", "map", "answer"]:
            schema = OlostepToolSchema(url="https://example.com", mode=mode)
            assert schema.mode == mode

    def test_schema_accepts_optional_parameters(self):
        """Test that schema accepts optional parameters."""
        from crewai_tools.tools.olostep_tool.olostep_tool import OlostepToolSchema

        schema = OlostepToolSchema(
            url="https://example.com",
            mode="scrape",
            formats=["markdown", "html"],
            wait_before_scraping=1000,
            country="US",
            parser="@olostep/google-search",
        )
        assert schema.formats == ["markdown", "html"]
        assert schema.wait_before_scraping == 1000
        assert schema.country == "US"
        assert schema.parser == "@olostep/google-search"


class TestOlostepCrawlSchema:
    """Tests for OlostepCrawlSchema validation."""

    def test_max_pages_validation_caps_at_1000(self):
        """Test that max_pages is capped at 1000."""
        from crewai_tools.tools.olostep_tool.olostep_tool import OlostepCrawlSchema

        schema = OlostepCrawlSchema(url="https://example.com", max_pages=5000)
        assert schema.max_pages == 1000

    def test_max_pages_validation_minimum_is_1(self):
        """Test that max_pages minimum is 1."""
        from crewai_tools.tools.olostep_tool.olostep_tool import OlostepCrawlSchema

        schema = OlostepCrawlSchema(url="https://example.com", max_pages=0)
        assert schema.max_pages == 1

    def test_default_max_pages_is_10(self):
        """Test that default max_pages is 10."""
        from crewai_tools.tools.olostep_tool.olostep_tool import OlostepCrawlSchema

        schema = OlostepCrawlSchema(url="https://example.com")
        assert schema.max_pages == 10


class TestOlostepToolSchemaMaxPages:
    """Tests for OlostepToolSchema max_pages validation."""

    def test_max_pages_validation_caps_at_1000(self):
        """Test that max_pages is capped at 1000 in main schema."""
        from crewai_tools.tools.olostep_tool.olostep_tool import OlostepToolSchema

        schema = OlostepToolSchema(url="https://example.com", mode="crawl", max_pages=5000)
        assert schema.max_pages == 1000

    def test_max_pages_validation_minimum_is_1(self):
        """Test that max_pages minimum is 1 in main schema."""
        from crewai_tools.tools.olostep_tool.olostep_tool import OlostepToolSchema

        schema = OlostepToolSchema(url="https://example.com", mode="crawl", max_pages=-5)
        assert schema.max_pages == 1

    def test_max_pages_validation_allows_none(self):
        """Test that max_pages can be None in main schema."""
        from crewai_tools.tools.olostep_tool.olostep_tool import OlostepToolSchema

        schema = OlostepToolSchema(url="https://example.com", mode="crawl", max_pages=None)
        assert schema.max_pages is None


class TestOlostepToolInitialization:
    """Tests for OlostepTool initialization."""

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_tool_initialization_defaults(self):
        """Test tool initializes with default values."""
        tool = OlostepTool()
        assert tool.name == "OlostepTool"
        assert tool.default_mode == "scrape"
        assert tool.default_formats == ["markdown"]
        assert tool.default_max_pages == 10
        assert tool.log_failures is True

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_tool_initialization_with_custom_values(self):
        """Test tool initializes with custom values."""
        tool = OlostepTool(
            api_key="custom-key",
            default_mode="crawl",
            default_formats=["html", "text"],
            default_max_pages=50,
            log_failures=False,
        )
        assert tool.api_key == "custom-key"
        assert tool.default_mode == "crawl"
        assert tool.default_formats == ["html", "text"]
        assert tool.default_max_pages == 50
        assert tool.log_failures is False

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_tool_has_package_dependencies(self):
        """Test tool declares olostep as package dependency."""
        tool = OlostepTool()
        assert "olostep" in tool.package_dependencies

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_tool_has_env_vars(self):
        """Test tool declares OLOSTEP_API_KEY env var."""
        tool = OlostepTool()
        env_var_names = [ev.name for ev in tool.env_vars]
        assert "OLOSTEP_API_KEY" in env_var_names


class TestOlostepToolScrapeMode:
    """Tests for OlostepTool scrape mode."""

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_scrape_returns_markdown_content(self, mock_client_class):
        """Test scrape mode returns markdown content."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.markdown = "# Test Content\n\nThis is test content."
        mock_result.llm_extract = None
        mock_result.parser_result = None
        mock_client.scrape.create.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        result = tool._run(url="https://example.com", mode="scrape")

        assert "Test Content" in result
        mock_client.scrape.create.assert_called_once()

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_scrape_with_parser(self, mock_client_class):
        """Test scrape mode with parser returns structured data."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.llm_extract = None
        mock_result.parser_result = {"title": "Test", "description": "Test desc"}
        mock_client.scrape.create.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        result = tool._run(
            url="https://google.com/search?q=test",
            mode="scrape",
            parser="@olostep/google-search",
        )

        assert "title" in result
        call_kwargs = mock_client.scrape.create.call_args[1]
        assert call_kwargs["parser"] == {"id": "@olostep/google-search"}

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_scrape_with_llm_extract(self, mock_client_class):
        """Test scrape mode with LLM extraction."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.llm_extract = {"event_name": "Concert", "date": "2024-01-01"}
        mock_result.parser_result = None
        mock_client.scrape.create.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        result = tool._run(
            url="https://example.com/event",
            mode="scrape",
            llm_extract_schema={"event_name": {"type": "string"}, "date": {"type": "string"}},
        )

        assert "event_name" in result
        assert "Concert" in result
        call_kwargs = mock_client.scrape.create.call_args[1]
        assert "llm_extract" in call_kwargs

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_scrape_with_country(self, mock_client_class):
        """Test scrape mode with country parameter."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.markdown = "Content from US"
        mock_result.llm_extract = None
        mock_result.parser_result = None
        mock_client.scrape.create.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        tool._run(url="https://example.com", mode="scrape", country="US")

        call_kwargs = mock_client.scrape.create.call_args[1]
        assert call_kwargs["country"] == "US"


class TestOlostepToolCrawlMode:
    """Tests for OlostepTool crawl mode."""

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_crawl_returns_job_info(self, mock_client_class):
        """Test crawl mode returns job information."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "crawl-123"
        mock_result.pages = []
        mock_client.crawl.start.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        result = tool._run(url="https://example.com", mode="crawl")

        assert "Crawl started successfully" in result
        assert "crawl-123" in result
        mock_client.crawl.start.assert_called_once()

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_crawl_respects_max_pages(self, mock_client_class):
        """Test crawl mode respects max_pages parameter."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "crawl-456"
        mock_result.pages = []
        mock_client.crawl.start.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        tool._run(url="https://example.com", mode="crawl", max_pages=25)

        call_kwargs = mock_client.crawl.start.call_args[1]
        assert call_kwargs["max_pages"] == 25

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_crawl_caps_max_pages_at_1000(self, mock_client_class):
        """Test crawl mode caps max_pages at 1000."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.id = "crawl-789"
        mock_result.pages = []
        mock_client.crawl.start.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        tool._run(url="https://example.com", mode="crawl", max_pages=5000)

        call_kwargs = mock_client.crawl.start.call_args[1]
        assert call_kwargs["max_pages"] == 1000


class TestOlostepToolMapMode:
    """Tests for OlostepTool map mode."""

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_map_returns_url_list(self, mock_client_class):
        """Test map mode returns list of URLs."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.urls = [
            "https://example.com/",
            "https://example.com/about",
            "https://example.com/contact",
        ]
        mock_client.sitemap.create.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        result = tool._run(url="https://example.com", mode="map")

        assert "Found 3 URLs" in result
        assert "example.com/about" in result
        mock_client.sitemap.create.assert_called_once()

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_map_handles_empty_result(self, mock_client_class):
        """Test map mode handles empty result gracefully."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.urls = []
        mock_result.links = None
        mock_client.sitemap.create.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        result = tool._run(url="https://example.com", mode="map")

        assert "No URLs found" in result


class TestOlostepToolAnswerMode:
    """Tests for OlostepTool answer mode."""

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_answer_returns_ai_response(self, mock_post):
        """Test answer mode returns AI response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": "The latest iPhone is iPhone 15"
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = OlostepTool(api_key="test-key")
        result = tool._run(
            mode="answer",
            task="What is the latest iPhone model?",
        )

        assert "iPhone" in result
        mock_post.assert_called_once()

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("requests.post")
    def test_answer_with_json_schema(self, mock_post):
        """Test answer mode with JSON schema."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": {"model": "iPhone 15", "release_year": 2023}
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tool = OlostepTool(api_key="test-key")
        result = tool._run(
            mode="answer",
            task="What is the latest iPhone model?",
            json_schema={"model": "", "release_year": 0},
        )

        assert "iPhone 15" in result
        assert "2023" in result


class TestOlostepToolErrorHandling:
    """Tests for OlostepTool error handling."""

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_scrape_requires_url(self):
        """Test scrape mode requires URL."""
        tool = OlostepTool(api_key="test-key")
        result = tool._run(mode="scrape")

        assert "Error" in result
        assert "URL is required" in result

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_crawl_requires_url(self):
        """Test crawl mode requires URL."""
        tool = OlostepTool(api_key="test-key")
        result = tool._run(mode="crawl")

        assert "Error" in result
        assert "URL is required" in result

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_map_requires_url(self):
        """Test map mode requires URL."""
        tool = OlostepTool(api_key="test-key")
        result = tool._run(mode="map")

        assert "Error" in result
        assert "URL is required" in result

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_answer_requires_task(self):
        """Test answer mode requires task."""
        tool = OlostepTool(api_key="test-key")
        result = tool._run(mode="answer")

        assert "Error" in result
        assert "task" in result.lower()

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_handles_api_error(self, mock_client_class):
        """Test tool handles API errors gracefully."""
        mock_client = MagicMock()
        mock_client.scrape.create.side_effect = Exception("API Error: Rate limit exceeded")
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        result = tool._run(url="https://example.com", mode="scrape")

        assert "error" in result.lower()
        assert "Rate limit" in result


class TestSpecializedTools:
    """Tests for specialized Olostep tools."""

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_scrape_tool_defaults_to_scrape_mode(self):
        """Test OlostepScrapeTool defaults to scrape mode."""
        tool = OlostepScrapeTool()
        assert tool.default_mode == "scrape"
        assert tool.name == "OlostepScrapeTool"

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_crawl_tool_defaults_to_crawl_mode(self):
        """Test OlostepCrawlTool defaults to crawl mode."""
        tool = OlostepCrawlTool()
        assert tool.default_mode == "crawl"
        assert tool.name == "OlostepCrawlTool"

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_map_tool_defaults_to_map_mode(self):
        """Test OlostepMapTool defaults to map mode."""
        tool = OlostepMapTool()
        assert tool.default_mode == "map"
        assert tool.name == "OlostepMapTool"

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    def test_answer_tool_defaults_to_answer_mode(self):
        """Test OlostepAnswerTool defaults to answer mode."""
        tool = OlostepAnswerTool()
        assert tool.default_mode == "answer"
        assert tool.name == "OlostepAnswerTool"


class TestOlostepToolParsers:
    """Tests for built-in parser support."""

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_google_search_parser(self, mock_client_class):
        """Test Google Search parser."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.llm_extract = None
        mock_result.parser_result = {
            "organic_results": [
                {"title": "Result 1", "link": "https://example.com"},
            ]
        }
        mock_client.scrape.create.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        result = tool._run(
            url="https://www.google.com/search?q=python",
            mode="scrape",
            parser="@olostep/google-search",
        )

        assert "organic_results" in result
        call_kwargs = mock_client.scrape.create.call_args[1]
        assert call_kwargs["parser"]["id"] == "@olostep/google-search"

    @patch.dict(os.environ, {"OLOSTEP_API_KEY": "test-api-key"})
    @patch("olostep.SyncOlostepClient")
    def test_linkedin_profile_parser(self, mock_client_class):
        """Test LinkedIn profile parser."""
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.llm_extract = None
        mock_result.parser_result = {
            "name": "John Doe",
            "headline": "Software Engineer",
        }
        mock_client.scrape.create.return_value = mock_result
        mock_client_class.return_value = mock_client

        tool = OlostepTool(api_key="test-key")
        result = tool._run(
            url="https://www.linkedin.com/in/johndoe",
            mode="scrape",
            parser="@olostep/linkedin-profile",
        )

        assert "John Doe" in result
        call_kwargs = mock_client.scrape.create.call_args[1]
        assert call_kwargs["parser"]["id"] == "@olostep/linkedin-profile"
