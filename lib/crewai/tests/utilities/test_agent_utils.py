"""Tests for agent utility functions."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.agent_utils import convert_tools_to_openai_schema, summarize_messages


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""

    expression: str = Field(description="Mathematical expression to evaluate")


class CalculatorTool(BaseTool):
    """A simple calculator tool for testing."""

    name: str = "calculator"
    description: str = "Perform mathematical calculations"
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Execute the calculation."""
        try:
            result = eval(expression)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class SearchInput(BaseModel):
    """Input schema for search tool."""

    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum number of results")


class SearchTool(BaseTool):
    """A search tool for testing."""

    name: str = "web_search"
    description: str = "Search the web for information"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, max_results: int = 10) -> str:
        """Execute the search."""
        return f"Search results for '{query}' (max {max_results})"


class NoSchemaTool(BaseTool):
    """A tool without an args schema for testing edge cases."""

    name: str = "simple_tool"
    description: str = "A simple tool with no schema"

    def _run(self, **kwargs: Any) -> str:
        """Execute the tool."""
        return "Simple tool executed"


class TestConvertToolsToOpenaiSchema:
    """Tests for convert_tools_to_openai_schema function."""

    def test_converts_single_tool(self) -> None:
        """Test converting a single tool to OpenAI schema."""
        tools = [CalculatorTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        assert len(schemas) == 1
        assert len(functions) == 1

        schema = schemas[0]
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculator"
        assert schema["function"]["description"] == "Perform mathematical calculations"
        assert "properties" in schema["function"]["parameters"]
        assert "expression" in schema["function"]["parameters"]["properties"]

    def test_converts_multiple_tools(self) -> None:
        """Test converting multiple tools to OpenAI schema."""
        tools = [CalculatorTool(), SearchTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        assert len(schemas) == 2
        assert len(functions) == 2

        # Check calculator
        calc_schema = next(s for s in schemas if s["function"]["name"] == "calculator")
        assert calc_schema["function"]["description"] == "Perform mathematical calculations"

        # Check search
        search_schema = next(s for s in schemas if s["function"]["name"] == "web_search")
        assert search_schema["function"]["description"] == "Search the web for information"
        assert "query" in search_schema["function"]["parameters"]["properties"]
        assert "max_results" in search_schema["function"]["parameters"]["properties"]

    def test_functions_dict_contains_callables(self) -> None:
        """Test that the functions dict maps names to callable run methods."""
        tools = [CalculatorTool(), SearchTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        assert "calculator" in functions
        assert "web_search" in functions
        assert callable(functions["calculator"])
        assert callable(functions["web_search"])

    def test_function_can_be_called(self) -> None:
        """Test that the returned function can be called."""
        tools = [CalculatorTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        result = functions["calculator"](expression="2 + 2")
        assert result == "4"

    def test_empty_tools_list(self) -> None:
        """Test with an empty tools list."""
        schemas, functions = convert_tools_to_openai_schema([])

        assert schemas == []
        assert functions == {}

    def test_schema_has_required_fields(self) -> None:
        """Test that the schema includes required fields information."""
        tools = [SearchTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        schema = schemas[0]
        params = schema["function"]["parameters"]

        # Should have required array
        assert "required" in params
        assert "query" in params["required"]

    def test_tool_without_args_schema(self) -> None:
        """Test converting a tool that doesn't have an args_schema."""
        # Create a minimal tool without args_schema
        class MinimalTool(BaseTool):
            name: str = "minimal"
            description: str = "A minimal tool"

            def _run(self) -> str:
                return "done"

        tools = [MinimalTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["function"]["name"] == "minimal"
        # Parameters should be empty dict or have minimal schema
        assert isinstance(schema["function"]["parameters"], dict)

    def test_schema_structure_matches_openai_format(self) -> None:
        """Test that the schema structure matches OpenAI's expected format."""
        tools = [CalculatorTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        schema = schemas[0]

        # Top level must have "type": "function"
        assert schema["type"] == "function"

        # Must have "function" key with nested structure
        assert "function" in schema
        func = schema["function"]

        # Function must have name and description
        assert "name" in func
        assert "description" in func
        assert isinstance(func["name"], str)
        assert isinstance(func["description"], str)

        # Parameters should be a valid JSON schema
        assert "parameters" in func
        params = func["parameters"]
        assert isinstance(params, dict)

    def test_removes_redundant_schema_fields(self) -> None:
        """Test that redundant title and description are removed from parameters."""
        tools = [CalculatorTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        # Title should be removed as it's redundant with function name
        assert "title" not in params

    def test_preserves_field_descriptions(self) -> None:
        """Test that field descriptions are preserved in the schema."""
        tools = [SearchTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        query_prop = params["properties"]["query"]

        # Field description should be preserved
        assert "description" in query_prop
        assert query_prop["description"] == "Search query"

    def test_preserves_default_values(self) -> None:
        """Test that default values are preserved in the schema."""
        tools = [SearchTool()]
        schemas, functions = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        max_results_prop = params["properties"]["max_results"]

        # Default value should be preserved
        assert "default" in max_results_prop
        assert max_results_prop["default"] == 10


class TestSummarizeMessages:
    """Tests for summarize_messages function."""

    def test_preserves_files_from_user_messages(self) -> None:
        """Test that files attached to user messages are preserved after summarization."""
        mock_files = {"image.png": MagicMock(), "doc.pdf": MagicMock()}
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Analyze this image", "files": mock_files},
            {"role": "assistant", "content": "I can see the image shows..."},
            {"role": "user", "content": "What about the colors?"},
        ]

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 1000
        mock_llm.call.return_value = "Summarized conversation about image analysis."

        mock_i18n = MagicMock()
        mock_i18n.slice.side_effect = lambda key: {
            "summarizer_system_message": "Summarize the following.",
            "summarize_instruction": "Summarize: {group}",
            "summary": "Summary: {merged_summary}",
        }.get(key, "")

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=mock_i18n,
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert "files" in messages[0]
        assert messages[0]["files"] == mock_files

    def test_merges_files_from_multiple_user_messages(self) -> None:
        """Test that files from multiple user messages are merged."""
        file1 = MagicMock()
        file2 = MagicMock()
        file3 = MagicMock()
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "First image", "files": {"img1.png": file1}},
            {"role": "assistant", "content": "I see the first image."},
            {"role": "user", "content": "Second image", "files": {"img2.png": file2, "doc.pdf": file3}},
            {"role": "assistant", "content": "I see the second image and document."},
        ]

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 1000
        mock_llm.call.return_value = "Summarized conversation."

        mock_i18n = MagicMock()
        mock_i18n.slice.side_effect = lambda key: {
            "summarizer_system_message": "Summarize the following.",
            "summarize_instruction": "Summarize: {group}",
            "summary": "Summary: {merged_summary}",
        }.get(key, "")

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=mock_i18n,
        )

        assert len(messages) == 1
        assert "files" in messages[0]
        assert messages[0]["files"] == {
            "img1.png": file1,
            "img2.png": file2,
            "doc.pdf": file3,
        }

    def test_works_without_files(self) -> None:
        """Test that summarization works when no files are attached."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 1000
        mock_llm.call.return_value = "A greeting exchange."

        mock_i18n = MagicMock()
        mock_i18n.slice.side_effect = lambda key: {
            "summarizer_system_message": "Summarize the following.",
            "summarize_instruction": "Summarize: {group}",
            "summary": "Summary: {merged_summary}",
        }.get(key, "")

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=mock_i18n,
        )

        assert len(messages) == 1
        assert "files" not in messages[0]

    def test_modifies_original_messages_list(self) -> None:
        """Test that the original messages list is modified in-place."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Response"},
            {"role": "user", "content": "Second message"},
        ]
        original_list_id = id(messages)

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 1000
        mock_llm.call.return_value = "Summary"

        mock_i18n = MagicMock()
        mock_i18n.slice.side_effect = lambda key: {
            "summarizer_system_message": "Summarize.",
            "summarize_instruction": "Summarize: {group}",
            "summary": "Summary: {merged_summary}",
        }.get(key, "")

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=mock_i18n,
        )

        assert id(messages) == original_list_id
        assert len(messages) == 1
