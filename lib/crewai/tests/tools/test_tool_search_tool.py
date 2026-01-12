"""Tests for the ToolSearchTool functionality."""

import json

import pytest
from pydantic import BaseModel

from crewai.tools import BaseTool, SearchStrategy, ToolSearchTool


class MockSearchTool(BaseTool):
    """A mock search tool for testing."""

    name: str = "Web Search"
    description: str = "Search the web for information on any topic."

    def _run(self, query: str) -> str:
        return f"Search results for: {query}"


class MockDatabaseTool(BaseTool):
    """A mock database tool for testing."""

    name: str = "Database Query"
    description: str = "Query a SQL database to retrieve data."

    def _run(self, query: str) -> str:
        return f"Database results for: {query}"


class MockScrapeTool(BaseTool):
    """A mock web scraping tool for testing."""

    name: str = "Web Scraper"
    description: str = "Scrape content from websites and extract text."

    def _run(self, url: str) -> str:
        return f"Scraped content from: {url}"


class MockEmailTool(BaseTool):
    """A mock email tool for testing."""

    name: str = "Send Email"
    description: str = "Send an email to a specified recipient."

    def _run(self, to: str, subject: str, body: str) -> str:
        return f"Email sent to {to}"


class MockCalculatorTool(BaseTool):
    """A mock calculator tool for testing."""

    name: str = "Calculator"
    description: str = "Perform mathematical calculations and arithmetic operations."

    def _run(self, expression: str) -> str:
        return f"Result: {eval(expression)}"


@pytest.fixture
def sample_tools() -> list[BaseTool]:
    """Create a list of sample tools for testing."""
    return [
        MockSearchTool(),
        MockDatabaseTool(),
        MockScrapeTool(),
        MockEmailTool(),
        MockCalculatorTool(),
    ]


@pytest.fixture
def tool_search(sample_tools: list[BaseTool]) -> ToolSearchTool:
    """Create a ToolSearchTool with sample tools."""
    return ToolSearchTool(tool_catalog=sample_tools)


class TestToolSearchToolCreation:
    """Tests for ToolSearchTool creation and initialization."""

    def test_create_tool_search_with_empty_catalog(self) -> None:
        """Test creating a ToolSearchTool with an empty catalog."""
        tool_search = ToolSearchTool()
        assert tool_search.name == "Tool Search"
        assert tool_search.tool_catalog == []
        assert tool_search.search_strategy == SearchStrategy.KEYWORD

    def test_create_tool_search_with_tools(self, sample_tools: list[BaseTool]) -> None:
        """Test creating a ToolSearchTool with a list of tools."""
        tool_search = ToolSearchTool(tool_catalog=sample_tools)
        assert len(tool_search.tool_catalog) == 5
        assert tool_search.get_catalog_size() == 5

    def test_create_tool_search_with_regex_strategy(
        self, sample_tools: list[BaseTool]
    ) -> None:
        """Test creating a ToolSearchTool with regex search strategy."""
        tool_search = ToolSearchTool(
            tool_catalog=sample_tools, search_strategy=SearchStrategy.REGEX
        )
        assert tool_search.search_strategy == SearchStrategy.REGEX

    def test_create_tool_search_with_custom_name(self) -> None:
        """Test creating a ToolSearchTool with a custom name."""
        tool_search = ToolSearchTool(name="My Tool Finder")
        assert tool_search.name == "My Tool Finder"


class TestToolSearchKeywordSearch:
    """Tests for keyword-based tool search."""

    def test_search_by_exact_name(self, tool_search: ToolSearchTool) -> None:
        """Test searching for a tool by its exact name."""
        result = tool_search._run("Web Search")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert len(result_data["tools"]) >= 1
        assert result_data["tools"][0]["name"] == "Web Search"

    def test_search_by_partial_name(self, tool_search: ToolSearchTool) -> None:
        """Test searching for a tool by partial name."""
        result = tool_search._run("Search")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert len(result_data["tools"]) >= 1
        tool_names = [t["name"] for t in result_data["tools"]]
        assert "Web Search" in tool_names

    def test_search_by_description_keyword(self, tool_search: ToolSearchTool) -> None:
        """Test searching for a tool by keyword in description."""
        result = tool_search._run("database")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert len(result_data["tools"]) >= 1
        tool_names = [t["name"] for t in result_data["tools"]]
        assert "Database Query" in tool_names

    def test_search_with_multiple_keywords(self, tool_search: ToolSearchTool) -> None:
        """Test searching with multiple keywords."""
        result = tool_search._run("web scrape content")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert len(result_data["tools"]) >= 1
        tool_names = [t["name"] for t in result_data["tools"]]
        assert "Web Scraper" in tool_names

    def test_search_no_results(self, tool_search: ToolSearchTool) -> None:
        """Test searching with a query that returns no results."""
        result = tool_search._run("xyznonexistent123abc")
        result_data = json.loads(result)

        assert result_data["status"] == "no_results"
        assert len(result_data["tools"]) == 0

    def test_search_max_results_limit(self, tool_search: ToolSearchTool) -> None:
        """Test that max_results limits the number of returned tools."""
        result = tool_search._run("tool", max_results=2)
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert len(result_data["tools"]) <= 2

    def test_search_empty_catalog(self) -> None:
        """Test searching with an empty tool catalog."""
        tool_search = ToolSearchTool()
        result = tool_search._run("search")
        result_data = json.loads(result)

        assert result_data["status"] == "error"
        assert "No tools available" in result_data["message"]


class TestToolSearchRegexSearch:
    """Tests for regex-based tool search."""

    def test_regex_search_simple_pattern(
        self, sample_tools: list[BaseTool]
    ) -> None:
        """Test regex search with a simple pattern."""
        tool_search = ToolSearchTool(
            tool_catalog=sample_tools, search_strategy=SearchStrategy.REGEX
        )
        result = tool_search._run("Web.*")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        tool_names = [t["name"] for t in result_data["tools"]]
        assert "Web Search" in tool_names or "Web Scraper" in tool_names

    def test_regex_search_case_insensitive(
        self, sample_tools: list[BaseTool]
    ) -> None:
        """Test that regex search is case insensitive."""
        tool_search = ToolSearchTool(
            tool_catalog=sample_tools, search_strategy=SearchStrategy.REGEX
        )
        result = tool_search._run("email")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        tool_names = [t["name"] for t in result_data["tools"]]
        assert "Send Email" in tool_names

    def test_regex_search_invalid_pattern_fallback(
        self, sample_tools: list[BaseTool]
    ) -> None:
        """Test that invalid regex patterns are escaped and still work."""
        tool_search = ToolSearchTool(
            tool_catalog=sample_tools, search_strategy=SearchStrategy.REGEX
        )
        result = tool_search._run("[invalid(regex")
        result_data = json.loads(result)

        assert result_data["status"] in ["success", "no_results"]


class TestToolSearchCustomSearch:
    """Tests for custom search function."""

    def test_custom_search_function(self, sample_tools: list[BaseTool]) -> None:
        """Test using a custom search function."""

        def custom_search(
            query: str, tools: list[BaseTool]
        ) -> list[BaseTool]:
            return [t for t in tools if "email" in t.name.lower()]

        tool_search = ToolSearchTool(
            tool_catalog=sample_tools, custom_search_fn=custom_search
        )
        result = tool_search._run("anything")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert len(result_data["tools"]) == 1
        assert result_data["tools"][0]["name"] == "Send Email"


class TestToolSearchCatalogManagement:
    """Tests for tool catalog management."""

    def test_add_tool(self, tool_search: ToolSearchTool) -> None:
        """Test adding a tool to the catalog."""
        initial_size = tool_search.get_catalog_size()

        class NewTool(BaseTool):
            name: str = "New Tool"
            description: str = "A new tool for testing."

            def _run(self) -> str:
                return "New tool result"

        tool_search.add_tool(NewTool())
        assert tool_search.get_catalog_size() == initial_size + 1

    def test_add_tools(self, tool_search: ToolSearchTool) -> None:
        """Test adding multiple tools to the catalog."""
        initial_size = tool_search.get_catalog_size()

        class NewTool1(BaseTool):
            name: str = "New Tool 1"
            description: str = "First new tool."

            def _run(self) -> str:
                return "Result 1"

        class NewTool2(BaseTool):
            name: str = "New Tool 2"
            description: str = "Second new tool."

            def _run(self) -> str:
                return "Result 2"

        tool_search.add_tools([NewTool1(), NewTool2()])
        assert tool_search.get_catalog_size() == initial_size + 2

    def test_remove_tool(self, tool_search: ToolSearchTool) -> None:
        """Test removing a tool from the catalog."""
        initial_size = tool_search.get_catalog_size()
        result = tool_search.remove_tool("Web Search")

        assert result is True
        assert tool_search.get_catalog_size() == initial_size - 1

    def test_remove_nonexistent_tool(self, tool_search: ToolSearchTool) -> None:
        """Test removing a tool that doesn't exist."""
        initial_size = tool_search.get_catalog_size()
        result = tool_search.remove_tool("Nonexistent Tool")

        assert result is False
        assert tool_search.get_catalog_size() == initial_size

    def test_list_tool_names(self, tool_search: ToolSearchTool) -> None:
        """Test listing all tool names in the catalog."""
        names = tool_search.list_tool_names()

        assert len(names) == 5
        assert "Web Search" in names
        assert "Database Query" in names
        assert "Web Scraper" in names
        assert "Send Email" in names
        assert "Calculator" in names


class TestToolSearchResultFormat:
    """Tests for the format of search results."""

    def test_result_contains_tool_info(self, tool_search: ToolSearchTool) -> None:
        """Test that search results contain complete tool information."""
        result = tool_search._run("Calculator")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        tool_info = result_data["tools"][0]

        assert "name" in tool_info
        assert "description" in tool_info
        assert "args_schema" in tool_info
        assert tool_info["name"] == "Calculator"

    def test_result_args_schema_format(self, tool_search: ToolSearchTool) -> None:
        """Test that args_schema is properly formatted."""
        result = tool_search._run("Email")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        tool_info = result_data["tools"][0]

        assert "args_schema" in tool_info
        args_schema = tool_info["args_schema"]
        assert isinstance(args_schema, dict)


class TestToolSearchIntegration:
    """Integration tests for ToolSearchTool."""

    def test_tool_search_as_base_tool(self, sample_tools: list[BaseTool]) -> None:
        """Test that ToolSearchTool works as a BaseTool."""
        tool_search = ToolSearchTool(tool_catalog=sample_tools)

        assert isinstance(tool_search, BaseTool)
        assert tool_search.name == "Tool Search"
        assert "search" in tool_search.description.lower()

    def test_tool_search_to_structured_tool(
        self, sample_tools: list[BaseTool]
    ) -> None:
        """Test converting ToolSearchTool to structured tool."""
        tool_search = ToolSearchTool(tool_catalog=sample_tools)
        structured = tool_search.to_structured_tool()

        assert structured.name == "Tool Search"
        assert structured.args_schema is not None

    def test_tool_search_run_method(self, tool_search: ToolSearchTool) -> None:
        """Test the run method of ToolSearchTool."""
        result = tool_search.run(query="search", max_results=3)

        assert isinstance(result, str)
        result_data = json.loads(result)
        assert "status" in result_data
        assert "tools" in result_data


class TestToolSearchScoring:
    """Tests for the keyword scoring algorithm."""

    def test_exact_name_match_scores_highest(
        self, sample_tools: list[BaseTool]
    ) -> None:
        """Test that exact name matches score higher than partial matches."""
        tool_search = ToolSearchTool(tool_catalog=sample_tools)
        result = tool_search._run("Web Search")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert result_data["tools"][0]["name"] == "Web Search"

    def test_name_match_scores_higher_than_description(
        self, sample_tools: list[BaseTool]
    ) -> None:
        """Test that name matches score higher than description matches."""
        tool_search = ToolSearchTool(tool_catalog=sample_tools)
        result = tool_search._run("Calculator")
        result_data = json.loads(result)

        assert result_data["status"] == "success"
        assert result_data["tools"][0]["name"] == "Calculator"
