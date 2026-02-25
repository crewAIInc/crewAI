"""Tests for agent utility functions."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from crewai.tools.base_tool import BaseTool
from crewai.utilities.agent_utils import (
    _asummarize_chunks,
    _estimate_token_count,
    _extract_summary_tags,
    _format_messages_for_summary,
    _split_messages_into_chunks,
    convert_tools_to_openai_schema,
    summarize_messages,
)


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


def _make_mock_i18n() -> MagicMock:
    """Create a mock i18n with the new structured prompt keys."""
    mock_i18n = MagicMock()
    mock_i18n.slice.side_effect = lambda key: {
        "summarizer_system_message": "You are a precise assistant that creates structured summaries.",
        "summarize_instruction": "Summarize the conversation:\n{conversation}",
        "summary": "<summary>\n{merged_summary}\n</summary>\nContinue the task.",
    }.get(key, "")
    return mock_i18n


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
        mock_llm.call.return_value = "<summary>Summarized conversation about image analysis.</summary>"

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        # System message preserved + summary message = 2
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        summary_msg = messages[1]
        assert summary_msg["role"] == "user"
        assert "files" in summary_msg
        assert summary_msg["files"] == mock_files

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
        mock_llm.call.return_value = "<summary>Summarized conversation.</summary>"

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
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
        mock_llm.call.return_value = "<summary>A greeting exchange.</summary>"

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
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
        mock_llm.call.return_value = "<summary>Summary</summary>"

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        assert id(messages) == original_list_id
        assert len(messages) == 1

    def test_preserves_system_messages(self) -> None:
        """Test that system messages are preserved and not summarized."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a research assistant."},
            {"role": "user", "content": "Find information about AI."},
            {"role": "assistant", "content": "I found several resources on AI."},
        ]

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 1000
        mock_llm.call.return_value = "<summary>User asked about AI, assistant found resources.</summary>"

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a research assistant."
        assert messages[1]["role"] == "user"

    def test_formats_conversation_with_role_labels(self) -> None:
        """Test that the LLM receives role-labeled conversation text."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "System prompt."},
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 1000
        mock_llm.call.return_value = "<summary>Greeting exchange.</summary>"

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        # Check what was passed to llm.call
        call_args = mock_llm.call.call_args[0][0]
        user_msg_content = call_args[1]["content"]
        assert "[USER]:" in user_msg_content
        assert "[ASSISTANT]:" in user_msg_content
        # System content should NOT appear in summarization input
        assert "System prompt." not in user_msg_content

    def test_extracts_summary_from_tags(self) -> None:
        """Test that <summary> tags are extracted from LLM response."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Do something."},
            {"role": "assistant", "content": "Done."},
        ]

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 1000
        mock_llm.call.return_value = "Here is the summary:\n<summary>The extracted summary content.</summary>\nExtra text."

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        assert "The extracted summary content." in messages[0]["content"]

    def test_handles_tool_messages(self) -> None:
        """Test that tool messages are properly formatted in summarization."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Search for Python."},
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "web_search", "arguments": '{"query": "Python"}'}}
            ]},
            {"role": "tool", "content": "Python is a programming language.", "name": "web_search"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 1000
        mock_llm.call.return_value = "<summary>User searched for Python info.</summary>"

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        # Verify the conversation text sent to LLM contains tool labels
        call_args = mock_llm.call.call_args[0][0]
        user_msg_content = call_args[1]["content"]
        assert "[TOOL_RESULT (web_search)]:" in user_msg_content

    def test_only_system_messages_no_op(self) -> None:
        """Test that only system messages results in no-op (no summarization)."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "Additional system instructions."},
        ]

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 1000

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        # No LLM call should have been made
        mock_llm.call.assert_not_called()
        # System messages should remain untouched
        assert len(messages) == 2
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["content"] == "Additional system instructions."


class TestFormatMessagesForSummary:
    """Tests for _format_messages_for_summary helper."""

    def test_skips_system_messages(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        result = _format_messages_for_summary(messages)
        assert "System prompt" not in result
        assert "[USER]: Hello" in result

    def test_formats_user_and_assistant(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        result = _format_messages_for_summary(messages)
        assert "[USER]: Question" in result
        assert "[ASSISTANT]: Answer" in result

    def test_formats_tool_messages(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "tool", "content": "Result data", "name": "search_tool"},
        ]
        result = _format_messages_for_summary(messages)
        assert "[TOOL_RESULT (search_tool)]:" in result
        assert "Result data" in result

    def test_handles_none_content_with_tool_calls(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "calculator", "arguments": "{}"}}
            ]},
        ]
        result = _format_messages_for_summary(messages)
        assert "[Called tools: calculator]" in result

    def test_handles_none_content_without_tool_calls(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": None},
        ]
        result = _format_messages_for_summary(messages)
        assert "[ASSISTANT]:" in result

    def test_handles_multimodal_content(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
            ]},
        ]
        result = _format_messages_for_summary(messages)
        assert "[USER]: Describe this image" in result

    def test_empty_messages(self) -> None:
        result = _format_messages_for_summary([])
        assert result == ""


class TestExtractSummaryTags:
    """Tests for _extract_summary_tags helper."""

    def test_extracts_content_from_tags(self) -> None:
        text = "Preamble\n<summary>The actual summary.</summary>\nPostamble"
        assert _extract_summary_tags(text) == "The actual summary."

    def test_handles_multiline_content(self) -> None:
        text = "<summary>\nLine 1\nLine 2\nLine 3\n</summary>"
        result = _extract_summary_tags(text)
        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_falls_back_when_no_tags(self) -> None:
        text = "Just a plain summary without tags."
        assert _extract_summary_tags(text) == text

    def test_handles_empty_string(self) -> None:
        assert _extract_summary_tags("") == ""

    def test_extracts_first_match(self) -> None:
        text = "<summary>First</summary> text <summary>Second</summary>"
        assert _extract_summary_tags(text) == "First"


class TestSplitMessagesIntoChunks:
    """Tests for _split_messages_into_chunks helper."""

    def test_single_chunk_when_under_limit(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        chunks = _split_messages_into_chunks(messages, max_tokens=1000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 2

    def test_splits_at_message_boundaries(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "A" * 100},  # ~25 tokens
            {"role": "assistant", "content": "B" * 100},  # ~25 tokens
            {"role": "user", "content": "C" * 100},  # ~25 tokens
        ]
        # max_tokens=30 should cause splits
        chunks = _split_messages_into_chunks(messages, max_tokens=30)
        assert len(chunks) == 3

    def test_excludes_system_messages(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "Hello"},
        ]
        chunks = _split_messages_into_chunks(messages, max_tokens=1000)
        assert len(chunks) == 1
        # The system message should not be in any chunk
        for chunk in chunks:
            for msg in chunk:
                assert msg.get("role") != "system"

    def test_empty_messages(self) -> None:
        chunks = _split_messages_into_chunks([], max_tokens=1000)
        assert chunks == []

    def test_only_system_messages(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": "System prompt"},
        ]
        chunks = _split_messages_into_chunks(messages, max_tokens=1000)
        assert chunks == []

    def test_handles_none_content(self) -> None:
        messages: list[dict[str, Any]] = [
            {"role": "assistant", "content": None},
            {"role": "user", "content": "Follow up"},
        ]
        chunks = _split_messages_into_chunks(messages, max_tokens=1000)
        assert len(chunks) == 1
        assert len(chunks[0]) == 2


class TestEstimateTokenCount:
    """Tests for _estimate_token_count helper."""

    def test_empty_string(self) -> None:
        assert _estimate_token_count("") == 0

    def test_short_string(self) -> None:
        assert _estimate_token_count("hello") == 1  # 5 // 4 = 1

    def test_longer_string(self) -> None:
        assert _estimate_token_count("a" * 100) == 25  # 100 // 4 = 25

    def test_approximation_is_conservative(self) -> None:
        # For English text, actual token count is typically lower than char/4
        text = "The quick brown fox jumps over the lazy dog."
        estimated = _estimate_token_count(text)
        assert estimated > 0
        assert estimated == len(text) // 4


class TestParallelSummarization:
    """Tests for parallel chunk summarization via asyncio."""

    def _make_messages_for_n_chunks(self, n: int) -> list[dict[str, Any]]:
        """Build a message list that will produce exactly *n* chunks.

        Each message has 400 chars (~100 tokens). With max_tokens=100 returned
        by the mock LLM, each message lands in its own chunk.
        """
        msgs: list[dict[str, Any]] = []
        for i in range(n):
            msgs.append({"role": "user", "content": f"msg-{i} " + "x" * 400})
        return msgs

    def test_multiple_chunks_use_acall(self) -> None:
        """When there are multiple chunks, summarize_messages should use
        llm.acall (parallel) instead of llm.call (sequential)."""
        messages = self._make_messages_for_n_chunks(3)

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 100  # force multiple chunks
        mock_llm.acall = AsyncMock(
            side_effect=[
                "<summary>Summary chunk 1</summary>",
                "<summary>Summary chunk 2</summary>",
                "<summary>Summary chunk 3</summary>",
            ]
        )

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        # acall should have been awaited once per chunk
        assert mock_llm.acall.await_count == 3
        # sync call should NOT have been used for chunk summarization
        mock_llm.call.assert_not_called()

    def test_single_chunk_uses_sync_call(self) -> None:
        """When there is only one chunk, summarize_messages should use
        the sync llm.call path (no async overhead)."""
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": "Short message"},
            {"role": "assistant", "content": "Short reply"},
        ]

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 100_000
        mock_llm.call.return_value = "<summary>Short summary</summary>"

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        mock_llm.call.assert_called_once()

    def test_parallel_results_preserve_order(self) -> None:
        """Summaries must appear in the same order as the original chunks,
        regardless of which async call finishes first."""
        messages = self._make_messages_for_n_chunks(3)

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 100

        # Simulate varying latencies — chunk 2 finishes before chunk 0
        async def _delayed_acall(msgs: Any, **kwargs: Any) -> str:
            user_content = msgs[1]["content"]
            if "msg-0" in user_content:
                await asyncio.sleep(0.05)
                return "<summary>Summary-A</summary>"
            elif "msg-1" in user_content:
                return "<summary>Summary-B</summary>"  # fastest
            else:
                await asyncio.sleep(0.02)
                return "<summary>Summary-C</summary>"

        mock_llm.acall = _delayed_acall

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        # The final summary message should have A, B, C in order
        summary_content = messages[-1]["content"]
        pos_a = summary_content.index("Summary-A")
        pos_b = summary_content.index("Summary-B")
        pos_c = summary_content.index("Summary-C")
        assert pos_a < pos_b < pos_c

    def test_asummarize_chunks_returns_ordered_results(self) -> None:
        """Direct test of the async helper _asummarize_chunks."""
        chunk_a: list[dict[str, Any]] = [{"role": "user", "content": "Chunk A"}]
        chunk_b: list[dict[str, Any]] = [{"role": "user", "content": "Chunk B"}]

        mock_llm = MagicMock()
        mock_llm.acall = AsyncMock(
            side_effect=[
                "<summary>Result A</summary>",
                "<summary>Result B</summary>",
            ]
        )

        results = asyncio.run(
            _asummarize_chunks(
                chunks=[chunk_a, chunk_b],
                llm=mock_llm,
                callbacks=[],
                i18n=_make_mock_i18n(),
            )
        )

        assert len(results) == 2
        assert results[0]["content"] == "Result A"
        assert results[1]["content"] == "Result B"

    @patch("crewai.utilities.agent_utils.is_inside_event_loop", return_value=True)
    def test_works_inside_existing_event_loop(self, _mock_loop: Any) -> None:
        """When called from inside a running event loop (e.g. a Flow),
        the ThreadPoolExecutor fallback should still work."""
        messages = self._make_messages_for_n_chunks(2)

        mock_llm = MagicMock()
        mock_llm.get_context_window_size.return_value = 100
        mock_llm.acall = AsyncMock(
            side_effect=[
                "<summary>Flow summary 1</summary>",
                "<summary>Flow summary 2</summary>",
            ]
        )

        summarize_messages(
            messages=messages,
            llm=mock_llm,
            callbacks=[],
            i18n=_make_mock_i18n(),
        )

        assert mock_llm.acall.await_count == 2
        # Verify the merged summary made it into messages
        assert "Flow summary 1" in messages[-1]["content"]
        assert "Flow summary 2" in messages[-1]["content"]


def _build_long_conversation() -> list[dict[str, Any]]:
    """Build a multi-turn conversation that produces multiple chunks at max_tokens=200.

    Each non-system message is ~100-140 estimated tokens (400-560 chars),
    so a max_tokens of 200 yields roughly 3 chunks from 6 messages.
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful research assistant.",
        },
        {
            "role": "user",
            "content": (
                "Tell me about the history of the Python programming language. "
                "Who created it, when was it first released, and what were the "
                "main design goals? Please provide a detailed overview covering "
                "the major milestones from its inception through Python 3."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Python was created by Guido van Rossum and first released in 1991. "
                "The main design goals were code readability and simplicity. Key milestones: "
                "Python 1.0 (1994) introduced functional programming tools like lambda and map. "
                "Python 2.0 (2000) added list comprehensions and garbage collection. "
                "Python 3.0 (2008) was a major backward-incompatible release that fixed "
                "fundamental design flaws. Python 2 reached end-of-life in January 2020."
            ),
        },
        {
            "role": "user",
            "content": (
                "What about the async/await features? When were they introduced "
                "and how do they compare to similar features in JavaScript and C#? "
                "Also explain the Global Interpreter Lock and its implications."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "Async/await was introduced in Python 3.5 (PEP 492, 2015). "
                "Unlike JavaScript which is single-threaded by design, Python's asyncio "
                "is an opt-in framework. C# introduced async/await in 2012 (C# 5.0) and "
                "was a major inspiration for Python's implementation. "
                "The GIL (Global Interpreter Lock) is a mutex that protects access to "
                "Python objects, preventing multiple threads from executing Python bytecodes "
                "simultaneously. This means CPU-bound multithreaded programs don't benefit "
                "from multiple cores. PEP 703 proposes making the GIL optional in CPython."
            ),
        },
        {
            "role": "user",
            "content": (
                "Explain the Python package ecosystem. How does pip work, what is PyPI, "
                "and what are virtual environments? Compare pip with conda and uv."
            ),
        },
        {
            "role": "assistant",
            "content": (
                "PyPI (Python Package Index) is the official repository hosting 400k+ packages. "
                "pip is the standard package installer that downloads from PyPI. "
                "Virtual environments (venv) create isolated Python installations to avoid "
                "dependency conflicts between projects. conda is a cross-language package manager "
                "popular in data science that can manage non-Python dependencies. "
                "uv is a new Rust-based tool that is 10-100x faster than pip and aims to replace "
                "pip, pip-tools, and virtualenv with a single unified tool."
            ),
        },
    ]


class TestParallelSummarizationVCR:
    """VCR-backed integration tests for parallel summarization.

    These tests use a real LLM but patch get_context_window_size to force
    multiple chunks, exercising the asyncio.gather + acall parallel path.

    To record cassettes:
        PYTEST_VCR_RECORD_MODE=all uv run pytest lib/crewai/tests/utilities/test_agent_utils.py::TestParallelSummarizationVCR -v
    """

    @pytest.mark.vcr()
    def test_parallel_summarize_openai(self) -> None:
        """Test that parallel summarization with gpt-4o-mini produces a valid summary."""
        from crewai.llm import LLM
        from crewai.utilities.i18n import I18N

        llm = LLM(model="gpt-4o-mini", temperature=0)
        i18n = I18N()
        messages = _build_long_conversation()

        original_system = messages[0]["content"]

        # Patch get_context_window_size to return 200 — forces multiple chunks
        with patch.object(type(llm), "get_context_window_size", return_value=200):
            # Verify we actually get multiple chunks with this window size
            non_system = [m for m in messages if m.get("role") != "system"]
            chunks = _split_messages_into_chunks(non_system, max_tokens=200)
            assert len(chunks) > 1, f"Expected multiple chunks, got {len(chunks)}"

            summarize_messages(
                messages=messages,
                llm=llm,
                callbacks=[],
                i18n=i18n,
            )

        # System message preserved
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == original_system

        # Summary produced as a user message
        summary_msg = messages[-1]
        assert summary_msg["role"] == "user"
        assert len(summary_msg["content"]) > 0

    @pytest.mark.vcr()
    def test_parallel_summarize_preserves_files(self) -> None:
        """Test that file references survive parallel summarization."""
        from crewai.llm import LLM
        from crewai.utilities.i18n import I18N

        llm = LLM(model="gpt-4o-mini", temperature=0)
        i18n = I18N()
        messages = _build_long_conversation()

        mock_file = MagicMock()
        messages[1]["files"] = {"report.pdf": mock_file}

        with patch.object(type(llm), "get_context_window_size", return_value=200):
            summarize_messages(
                messages=messages,
                llm=llm,
                callbacks=[],
                i18n=i18n,
            )

        summary_msg = messages[-1]
        assert summary_msg["role"] == "user"
        assert "files" in summary_msg
        assert "report.pdf" in summary_msg["files"]
