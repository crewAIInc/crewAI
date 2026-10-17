"""Tests for agent utility functions."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Literal, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field, ImportString

from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    clear_after_tool_call_hooks,
    clear_before_tool_call_hooks,
    register_after_tool_call_hook,
)
from crewai.agents.parser import AgentFinish
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    ContextWindowSummarizationCompletedEvent,
    ContextWindowSummarizationFailedEvent,
    ContextWindowSummarizationStartedEvent,
)
from crewai.hooks.dispatch import HookAborted
from crewai.llms.base_llm import LLMCallBlockedError
from crewai.tools.base_tool import BaseTool
from crewai.llms.context_window import CONTEXT_WINDOW_USAGE_RATIO
from crewai.utilities.agent_utils import (
    _summarize_context_before_llm_call,
    aget_llm_response,
    get_llm_response,
    LLMContextLengthExceededError,
    convert_tools_to_openai_schema,
    handle_max_iterations_exceeded,
    execute_single_native_tool_call,
    extract_tool_call_info,
    is_tool_call_list,
    NativeToolCallResult,
    parse_tool_call_args,
)
from crewai.utilities.message_utils import format_message_for_llm, message_content_text
from crewai_core.printer import Printer
from crewai.utilities.i18n import I18N_DEFAULT



def test_summarizes_context_before_llm_call() -> None:
    messages = [
        {"role": "user", "content": "earlier context"},
        {"role": "assistant", "content": "earlier response"},
        {"role": "user", "content": "large context"},
    ]
    llm = MagicMock()

    with patch.object(llm, "summarize_messages") as summarize:
        _summarize_context_before_llm_call(
            llm=llm,
            messages=messages,
            callbacks=[],
            printer=Printer(),
            verbose=False,
        )

    summarize.assert_called_once_with(messages, [], False)


def test_context_summarization_shows_triggered_and_completed_panels() -> None:
    messages = [
        {"role": "user", "content": "earlier context"},
        {"role": "assistant", "content": "earlier response"},
        {"role": "user", "content": "large context"},
    ]
    llm = MagicMock()
    llm.get_input_tokens_count.side_effect = [1_000, 250]

    with (
        patch.object(llm, "summarize_messages"),
        patch("crewai.utilities.agent_utils.console.print") as print_panel,
    ):
        _summarize_context_before_llm_call(
            llm=llm,
            messages=messages,
            callbacks=[],
            printer=Printer(),
            verbose=True,
        )

    assert print_panel.call_count == 2
    assert "triggered" in str(print_panel.call_args_list[0].args[0].title).lower()
    assert "completed" in str(print_panel.call_args_list[1].args[0].title).lower()
    assert "1,000 tokens before compaction" in str(
        print_panel.call_args_list[0].args[0].renderable
    )
    assert "1,000 → 250 tokens (75.0% reduction)" in str(
        print_panel.call_args_list[1].args[0].renderable
    )


def test_context_summarization_emits_lifecycle_events_with_llm_agent_and_task() -> None:
    messages = [
        {"role": "user", "content": "earlier context"},
        {"role": "assistant", "content": "earlier response"},
        {"role": "user", "content": "large context"},
    ]
    llm = MagicMock(model="gpt-4o-mini")
    task = SimpleNamespace(id="task-id", name="Research", description="Research")
    agent = SimpleNamespace(id="agent-id", role="Researcher")

    with (
        patch.object(llm, "summarize_messages"),
        patch.object(crewai_event_bus, "emit") as emit,
    ):
        _summarize_context_before_llm_call(
            llm=llm,
            messages=messages,
            callbacks=[],
            printer=Printer(),
            verbose=False,
            from_task=task,
            from_agent=agent,
        )

    events = [call.kwargs["event"] for call in emit.call_args_list]
    assert len(events) == 2
    assert isinstance(events[0], ContextWindowSummarizationStartedEvent)
    assert isinstance(events[1], ContextWindowSummarizationCompletedEvent)
    for event in events:
        assert event.model == "gpt-4o-mini"
        assert event.agent_id == "agent-id"
        assert event.agent_role == "Researcher"
        assert event.task_id == "task-id"
        assert event.task_name == "Research"


def test_context_summarization_emits_failed_event_when_compaction_fails() -> None:
    messages = [
        {"role": "user", "content": "earlier context"},
        {"role": "assistant", "content": "earlier response"},
        {"role": "user", "content": "large context"},
    ]
    llm = MagicMock(model="gpt-4o-mini")

    with (
        patch.object(
            llm, "summarize_messages", side_effect=RuntimeError("summarization failed")
        ),
        patch.object(crewai_event_bus, "emit") as emit,
        pytest.raises(RuntimeError, match="summarization failed"),
    ):
        _summarize_context_before_llm_call(
            llm=llm,
            messages=messages,
            callbacks=[],
            printer=Printer(),
            verbose=False,
        )

    events = [call.kwargs["event"] for call in emit.call_args_list]
    assert len(events) == 2
    assert isinstance(events[0], ContextWindowSummarizationStartedEvent)
    assert isinstance(events[1], ContextWindowSummarizationFailedEvent)
    assert events[1].error == "summarization failed"


def test_get_llm_response_summarizes_before_the_model_call() -> None:
    messages = [
        {"role": "user", "content": "earlier context"},
        {"role": "assistant", "content": "earlier response"},
        {"role": "user", "content": "large context"},
    ]
    executor = SimpleNamespace(
        messages=messages,
        respect_context_window=True,
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
    )
    llm = MagicMock()
    llm.is_context_window_overflowing_threshold.return_value = True
    llm.call.return_value = "response"
    events: list[str] = []

    with patch.object(llm, "summarize_messages") as summarize:
        summarize.side_effect = lambda *_args, **_kwargs: events.append("summarize")
        llm.call.side_effect = lambda *_args, **_kwargs: events.append("call") or "response"

        assert (
            get_llm_response(
                llm=llm,
                messages=messages,
                callbacks=[],
                printer=Printer(),
                executor_context=executor,
                verbose=False,
            )
            == "response"
        )

    assert events == ["summarize", "call"]


def test_get_llm_response_preserves_a_single_pending_user_request() -> None:
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "large context"},
    ]
    executor = SimpleNamespace(
        messages=messages,
        respect_context_window=True,
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
    )
    llm = MagicMock()
    llm.is_context_window_overflowing_threshold.return_value = True
    llm.call.return_value = "response"

    with patch.object(llm, "summarize_messages") as summarize:
        assert (
            get_llm_response(
                llm=llm,
                messages=messages,
                callbacks=[],
                printer=Printer(),
                executor_context=executor,
                verbose=False,
            )
            == "response"
        )

    summarize.assert_not_called()
    assert llm.call.call_args.args[0] == messages


@pytest.mark.parametrize(
    "failure_source",
    ["preflight", "summarization"],
)
def test_get_llm_response_continues_when_proactive_compaction_fails(
    failure_source: str,
) -> None:
    messages = [
        {"role": "user", "content": "earlier context"},
        {"role": "assistant", "content": "earlier response"},
        {"role": "user", "content": "pending request"},
    ]
    executor = SimpleNamespace(
        messages=messages,
        respect_context_window=True,
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
    )
    llm = MagicMock()
    llm.call.return_value = "response"
    if failure_source == "preflight":
        llm.is_context_window_overflowing_threshold.side_effect = RuntimeError(
            "token counting failed"
        )
        summarize = patch.object(llm, "summarize_messages")
    else:
        llm.is_context_window_overflowing_threshold.return_value = True
        summarize = patch.object(
            llm, "summarize_messages", side_effect=RuntimeError("summarization failed")
        )

    with summarize:
        assert (
            get_llm_response(
                llm=llm,
                messages=messages,
                callbacks=[],
                printer=Printer(),
                executor_context=executor,
                verbose=False,
            )
            == "response"
        )

    llm.call.assert_called_once()


@pytest.mark.parametrize(
    "error", [HookAborted("blocked"), LLMCallBlockedError("blocked")])
def test_get_llm_response_propagates_explicit_proactive_compaction_blocks(
    error: Exception,
) -> None:
    messages = [{"role": "user", "content": "pending request"}]
    executor = SimpleNamespace(
        messages=messages,
        respect_context_window=True,
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
    )
    llm = MagicMock()
    llm.is_context_window_overflowing_threshold.side_effect = error

    with pytest.raises(type(error)):
        get_llm_response(
            llm=llm,
            messages=messages,
            callbacks=[],
            printer=Printer(),
            executor_context=executor,
            verbose=False,
        )

    llm.call.assert_not_called()


def test_get_llm_response_skips_preflight_when_context_respect_is_disabled() -> None:
    messages = [{"role": "user", "content": "large context"}]
    executor = SimpleNamespace(
        messages=messages,
        respect_context_window=False,
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
    )
    llm = MagicMock()
    llm.call.return_value = "response"

    assert (
        get_llm_response(
            llm=llm,
            messages=messages,
            callbacks=[],
            printer=Printer(),
            executor_context=executor,
            verbose=False,
        )
        == "response"
    )

    llm.is_context_window_overflowing_threshold.assert_not_called()


@pytest.mark.asyncio
async def test_aget_llm_response_summarizes_before_the_model_call() -> None:
    messages = [
        {"role": "user", "content": "earlier context"},
        {"role": "assistant", "content": "earlier response"},
        {"role": "user", "content": "large context"},
    ]
    executor = SimpleNamespace(
        messages=messages,
        respect_context_window=True,
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
    )
    llm = MagicMock()
    llm.is_context_window_overflowing_threshold.return_value = True
    llm.acall = AsyncMock(return_value="response")

    with patch.object(llm, "summarize_messages") as summarize:
        assert (
            await aget_llm_response(
                llm=llm,
                messages=messages,
                callbacks=[],
                printer=Printer(),
                executor_context=executor,
                verbose=False,
            )
            == "response"
        )

    summarize.assert_called_once()
    llm.acall.assert_awaited_once()


@pytest.mark.asyncio
async def test_aget_llm_response_offloads_proactive_compaction() -> None:
    messages = [
        {"role": "user", "content": "earlier context"},
        {"role": "assistant", "content": "earlier response"},
        {"role": "user", "content": "pending request"},
    ]
    executor = SimpleNamespace(
        messages=messages,
        respect_context_window=True,
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
    )
    llm = MagicMock()
    llm.is_context_window_overflowing_threshold.return_value = False
    llm.acall = AsyncMock(return_value="response")

    with patch(
        "crewai.utilities.agent_utils.asyncio.to_thread",
        wraps=asyncio.to_thread,
    ) as to_thread:
        assert (
            await aget_llm_response(
                llm=llm,
                messages=messages,
                callbacks=[],
                printer=Printer(),
                executor_context=executor,
                verbose=False,
            )
            == "response"
        )

    assert to_thread.await_count == 1
    assert to_thread.call_args.args[0].__name__ == "_proactively_summarize_context"
    llm.is_context_window_overflowing_threshold.assert_called_once()
    llm.acall.assert_awaited_once()


@pytest.mark.asyncio
async def test_aget_llm_response_skips_preflight_when_context_respect_is_disabled() -> None:
    messages = [{"role": "user", "content": "large context"}]
    executor = SimpleNamespace(
        messages=messages,
        respect_context_window=False,
        before_llm_call_hooks=[],
        after_llm_call_hooks=[],
    )
    llm = MagicMock()
    llm.acall = AsyncMock(return_value="response")

    assert (
        await aget_llm_response(
            llm=llm,
            messages=messages,
            callbacks=[],
            printer=Printer(),
            executor_context=executor,
            verbose=False,
        )
        == "response"
    )

    llm.is_context_window_overflowing_threshold.assert_not_called()

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
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

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
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

        assert len(schemas) == 2
        assert len(functions) == 2

        calc_schema = next(s for s in schemas if s["function"]["name"] == "calculator")
        assert calc_schema["function"]["description"] == "Perform mathematical calculations"

        search_schema = next(s for s in schemas if s["function"]["name"] == "web_search")
        assert search_schema["function"]["description"] == "Search the web for information"
        assert "query" in search_schema["function"]["parameters"]["properties"]
        assert "max_results" in search_schema["function"]["parameters"]["properties"]

    def test_functions_dict_contains_callables(self) -> None:
        """Test that the functions dict maps names to callable run methods."""
        tools = [CalculatorTool(), SearchTool()]
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

        assert "calculator" in functions
        assert "web_search" in functions
        assert callable(functions["calculator"])
        assert callable(functions["web_search"])

    def test_function_can_be_called(self) -> None:
        """Test that the returned function can be called."""
        tools = [CalculatorTool()]
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

        result = functions["calculator"](expression="2 + 2")
        assert result == "4"

    def test_empty_tools_list(self) -> None:
        """Test with an empty tools list."""
        schemas, functions, _ = convert_tools_to_openai_schema([])

        assert schemas == []
        assert functions == {}

    def test_schema_has_required_fields(self) -> None:
        """Test that the schema includes required fields information."""
        tools = [SearchTool()]
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

        schema = schemas[0]
        params = schema["function"]["parameters"]

        assert "required" in params
        assert "query" in params["required"]

    def test_tool_without_args_schema(self) -> None:
        """Test converting a tool that doesn't have an args_schema."""
        class MinimalTool(BaseTool):
            name: str = "minimal"
            description: str = "A minimal tool"

            def _run(self) -> str:
                return "done"

        tools = [MinimalTool()]
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

        assert len(schemas) == 1
        schema = schemas[0]
        assert schema["function"]["name"] == "minimal"
        # Parameters should be empty dict or have minimal schema
        assert isinstance(schema["function"]["parameters"], dict)

    def test_schema_structure_matches_openai_format(self) -> None:
        """Test that the schema structure matches OpenAI's expected format."""
        tools = [CalculatorTool()]
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

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
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        # Title should be removed as it's redundant with function name
        assert "title" not in params

    def test_preserves_field_descriptions(self) -> None:
        """Test that field descriptions are preserved in the schema."""
        tools = [SearchTool()]
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        query_prop = params["properties"]["query"]

        # Field description should be preserved
        assert "description" in query_prop
        assert query_prop["description"] == "Search query"

    def test_preserves_default_values(self) -> None:
        """Test that default values are preserved in the schema."""
        tools = [SearchTool()]
        schemas, functions, _ = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        max_results_prop = params["properties"]["max_results"]

        # Default value should be preserved
        assert "default" in max_results_prop
        assert max_results_prop["default"] == 10


class MCPStyleInput(BaseModel):
    """Input schema mimicking an MCP tool with optional fields."""

    query: str = Field(description="Search query")
    filter_type: Optional[Literal["internal", "user"]] = Field(
        default=None, description="Filter type"
    )
    page_id: Optional[str] = Field(
        default=None, description="Page UUID"
    )


class MCPStyleTool(BaseTool):
    """A tool mimicking MCP tool schemas with optional fields."""

    name: str = "mcp_search"
    description: str = "Search with optional filters"
    args_schema: type[BaseModel] = MCPStyleInput

    def _run(self, **kwargs: Any) -> str:
        return "result"


class TestOptionalFieldsPreserveNull:
    """Tests that optional tool fields preserve null in the schema."""

    def test_optional_string_allows_null(self) -> None:
        """Optional[str] fields should include null in the schema so the LLM
        can send null instead of being forced to guess a value."""
        tools = [MCPStyleTool()]
        schemas, _, _ = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        page_id_prop = params["properties"]["page_id"]

        assert "anyOf" in page_id_prop
        type_options = [opt.get("type") for opt in page_id_prop["anyOf"]]
        assert "string" in type_options
        assert "null" in type_options

    def test_optional_literal_allows_null(self) -> None:
        """Optional[Literal[...]] fields should include null."""
        tools = [MCPStyleTool()]
        schemas, _, _ = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        filter_prop = params["properties"]["filter_type"]

        assert "anyOf" in filter_prop
        has_null = any(opt.get("type") == "null" for opt in filter_prop["anyOf"])
        assert has_null

    def test_required_field_stays_non_null(self) -> None:
        """Required fields without Optional should NOT have null."""
        tools = [MCPStyleTool()]
        schemas, _, _ = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        query_prop = params["properties"]["query"]

        assert query_prop.get("type") == "string"
        assert "anyOf" not in query_prop

    def test_all_fields_in_required_for_strict_mode(self) -> None:
        """All fields (including optional) must be in required for strict mode."""
        tools = [MCPStyleTool()]
        schemas, _, _ = convert_tools_to_openai_schema(tools)

        params = schemas[0]["function"]["parameters"]
        assert "query" in params["required"]
        assert "filter_type" in params["required"]
        assert "page_id" in params["required"]


class TestMessageContentText:
    """Tests for message_content_text helper."""

    def test_string_content(self) -> None:
        msg: dict[str, Any] = {"role": "user", "content": "hello"}
        assert message_content_text(msg) == "hello"

    def test_none_content(self) -> None:
        msg: dict[str, Any] = {"role": "assistant", "content": None}
        assert message_content_text(msg) == ""

    def test_list_content_yields_its_text(self) -> None:
        """A parts list used to collapse to its Python repr."""
        msg: dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}],
        }
        assert message_content_text(msg) == "first second"

    @pytest.mark.parametrize(
        "bad_text", [123, None, {"nested": "x"}, ["a"]], ids=str
    )
    def test_a_non_string_text_block_does_not_raise(self, bad_text: Any) -> None:
        """Blocks are `dict[str, Any]` from a model, so `text` may be anything."""
        msg: dict[str, Any] = {
            "role": "user",
            "content": [{"type": "text", "text": bad_text}],
        }

        assert message_content_text(msg) == "[multimodal content]"

    def test_a_usable_text_block_survives_a_malformed_sibling(self) -> None:
        msg: dict[str, Any] = {
            "role": "user",
            "content": [
                {"type": "text", "text": {"nested": "x"}},
                {"type": "text", "text": "real text"},
            ],
        }

        assert message_content_text(msg) == "real text"

    def test_list_content_without_text_is_named_not_repr(self) -> None:
        msg: dict[str, Any] = {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://x/y.png"}}],
        }

        text = message_content_text(msg)

        assert text == "[multimodal content]"
        assert "image_url" not in text


class TestIsToolCallListResponsesApiShape:
    """Regression tests: OpenAI Responses API tool-call dicts must be recognized.

    Responses API function_call output items are flat dicts shaped
    {"id", "name", "arguments"} - no nested "function" key, and "arguments"
    instead of Anthropic/Bedrock-style "input".
    """

    def test_responses_api_dict_is_recognized_as_tool_call(self) -> None:
        response = [
            {
                "id": "call_abc123",
                "name": "fetch_page",
                "arguments": '{"url": "https://example.com"}',
            }
        ]
        assert is_tool_call_list(response) is True

    def test_plain_text_answer_not_misclassified(self) -> None:
        assert is_tool_call_list(["just a string, not a tool call"]) is False

    def test_empty_list_returns_false(self) -> None:
        assert is_tool_call_list([]) is False

    def test_chat_completions_style_still_recognized(self) -> None:
        response = [{"function": {"name": "fetch_page", "arguments": "{}"}}]
        assert is_tool_call_list(response) is True

    def test_bedrock_anthropic_style_still_recognized(self) -> None:
        response = [{"name": "fetch_page", "input": {"url": "https://example.com"}}]
        assert is_tool_call_list(response) is True


class TestExtractToolCallInfoResponsesApiShape:
    """Regression tests: extract_tool_call_info must parse Responses API dicts."""

    def test_responses_api_dict_extracts_real_arguments(self) -> None:
        tool_call = {
            "id": "call_abc123",
            "name": "fetch_page",
            "arguments": '{"url": "https://example.com"}',
        }
        result = extract_tool_call_info(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert call_id == "call_abc123"
        assert func_name == "fetch_page"
        assert func_args == '{"url": "https://example.com"}'

    def test_responses_api_dict_does_not_return_empty_args(self) -> None:
        tool_call = {
            "id": "call_xyz",
            "name": "fetch_page",
            "arguments": '{"url": "https://example.com"}',
        }
        _, _, func_args = extract_tool_call_info(tool_call)
        assert func_args != {}

    def test_bedrock_anthropic_style_still_uses_input(self) -> None:
        tool_call = {"name": "fetch_page", "input": {"url": "https://example.com"}}
        _, func_name, func_args = extract_tool_call_info(tool_call)
        assert func_name == "fetch_page"
        assert func_args == {"url": "https://example.com"}

    def test_chat_completions_style_still_uses_nested_function(self) -> None:
        tool_call = {
            "id": "call_1",
            "function": {"name": "fetch_page", "arguments": "{}"},
        }
        _, func_name, func_args = extract_tool_call_info(tool_call)
        assert func_name == "fetch_page"
        assert func_args == "{}"

    def test_non_dict_unrecognized_shape_returns_none(self) -> None:
        assert extract_tool_call_info("just a string") is None

    def test_unrecognized_dict_shape_returns_empty_name_and_args(self) -> None:
        call_id, func_name, func_args = extract_tool_call_info({"unrelated": "data"})
        assert func_name == ""
        assert func_args == {}


class TestParseToolCallArgs:
    """Unit tests for parse_tool_call_args."""

    def test_valid_json_string_returns_dict(self) -> None:
        args_dict, error = parse_tool_call_args('{"code": "print(1)"}', "run_code", "call_1")
        assert error is None
        assert args_dict == {"code": "print(1)"}

    def test_malformed_json_returns_error_dict(self) -> None:
        args_dict, error = parse_tool_call_args('{"code": "print("hi")"}', "run_code", "call_1")
        assert args_dict is None
        assert error is not None
        assert error["call_id"] == "call_1"
        assert error["func_name"] == "run_code"
        assert error["from_cache"] is False
        assert "Failed to parse tool arguments as JSON" in error["result"]
        assert "run_code" in error["result"]

    def test_malformed_json_preserves_original_tool(self) -> None:
        mock_tool = object()
        _, error = parse_tool_call_args("{bad}", "my_tool", "call_2", original_tool=mock_tool)
        assert error is not None
        assert error["original_tool"] is mock_tool

    def test_malformed_json_original_tool_defaults_to_none(self) -> None:
        _, error = parse_tool_call_args("{bad}", "my_tool", "call_3")
        assert error is not None
        assert error["original_tool"] is None

    def test_dict_input_returned_directly(self) -> None:
        func_args = {"code": "x = 42"}
        args_dict, error = parse_tool_call_args(func_args, "run_code", "call_4")
        assert error is None
        assert args_dict == {"code": "x = 42"}

    def test_empty_dict_input_returned_directly(self) -> None:
        args_dict, error = parse_tool_call_args({}, "run_code", "call_5")
        assert error is None
        assert args_dict == {}

    def test_valid_json_with_nested_values(self) -> None:
        args_dict, error = parse_tool_call_args(
            '{"query": "hello", "options": {"limit": 10}}', "search", "call_6"
        )
        assert error is None
        assert args_dict == {"query": "hello", "options": {"limit": 10}}

    def test_error_result_has_correct_keys(self) -> None:
        _, error = parse_tool_call_args("{bad json}", "tool", "call_7")
        assert error is not None
        assert set(error.keys()) == {
            "call_id",
            "func_name",
            "result",
            "from_cache",
            "original_tool",
            "tool_failure",
        }


class TestExecuteSingleNativeToolCall:
    """Tests for execute_single_native_tool_call."""

    def test_typed_tool_output_is_json_agent_text(self) -> None:
        clear_before_tool_call_hooks()
        clear_after_tool_call_hooks()

        class SearchOutput(BaseModel):
            query: str
            score: float

        class TypedSearchTool(BaseTool):
            name: str = "typed_search"
            description: str = "Search for a query"
            result_schema: type[BaseModel] = SearchOutput

            def _run(self, query: str) -> SearchOutput:
                return SearchOutput(query=query, score=0.9)

        tool = TypedSearchTool()
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "typed_search"
        tool_call.function.arguments = '{"query": "crew"}'

        result = execute_single_native_tool_call(
            tool_call,
            available_functions={"typed_search": tool._run},
            original_tools=[tool],
            structured_tools=[tool.to_structured_tool()],
            tools_handler=None,
            agent=None,
            task=None,
            crew=None,
            event_source=MagicMock(),
            printer=None,
            verbose=False,
        )

        assert json.loads(result.result) == {"query": "crew", "score": 0.9}
        assert json.loads(result.tool_message["content"]) == {
            "query": "crew",
            "score": 0.9,
        }

    def test_custom_agent_output_formatter_is_used_from_structured_tool(
        self,
    ) -> None:
        clear_before_tool_call_hooks()
        clear_after_tool_call_hooks()

        class SearchOutput(BaseModel):
            query: str
            score: float

        class MarkdownSearchTool(BaseTool):
            name: str = "markdown_search"
            description: str = "Search for a query"
            result_schema: type[BaseModel] = SearchOutput

            def _run(self, query: str) -> SearchOutput:
                return SearchOutput(query=query, score=0.9)

            def format_output_for_agent(self, raw_result: Any) -> str:
                result = self.result_schema.model_validate(raw_result)
                return f"### {result.query}\n\nScore: **{result.score}**"

        tool = MarkdownSearchTool()
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "markdown_search"
        tool_call.function.arguments = '{"query": "crew"}'

        result = execute_single_native_tool_call(
            tool_call,
            available_functions={"markdown_search": tool._run},
            original_tools=[],
            structured_tools=[tool.to_structured_tool()],
            tools_handler=None,
            agent=None,
            task=None,
            crew=None,
            event_source=MagicMock(),
            printer=None,
            verbose=False,
        )

        assert result.result == "### crew\n\nScore: **0.9**"
        assert result.tool_message["content"] == "### crew\n\nScore: **0.9**"

    def test_after_hook_includes_raw_tool_result_for_typed_output(self) -> None:
        clear_after_tool_call_hooks()

        class SearchOutput(BaseModel):
            query: str
            score: float

        class TypedSearchTool(BaseTool):
            name: str = "typed_search"
            description: str = "Search for a query"
            result_schema: type[BaseModel] = SearchOutput

            def _run(self, query: str) -> SearchOutput:
                return SearchOutput(query=query, score=0.9)

        seen_results: list[tuple[str | None, object]] = []

        def after_hook(context: ToolCallHookContext) -> None:
            seen_results.append((context.tool_result, context.raw_tool_result))

        tool = TypedSearchTool()
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "typed_search"
        tool_call.function.arguments = '{"query": "crew"}'

        register_after_tool_call_hook(after_hook)
        try:
            result = execute_single_native_tool_call(
                tool_call,
                available_functions={"typed_search": tool._run},
                original_tools=[tool],
                structured_tools=[tool.to_structured_tool()],
                tools_handler=None,
                agent=None,
                task=None,
                crew=None,
                event_source=MagicMock(),
                printer=None,
                verbose=False,
            )
        finally:
            clear_after_tool_call_hooks()

        assert json.loads(result.result) == {"query": "crew", "score": 0.9}
        assert seen_results == [
            ('{"query":"crew","score":0.9}', SearchOutput(query="crew", score=0.9))
        ]

    def test_result_as_answer_false_on_tool_error(self) -> None:
        """When a tool with result_as_answer=True raises, result_as_answer must be False.

        Regression test for https://github.com/crewAIInc/crewAI/issues/5156
        """
        from unittest.mock import MagicMock

        class FailingTool(BaseTool):
            name: str = "failing_tool"
            description: str = "A tool that always fails"
            result_as_answer: bool = True

            def _run(self, **kwargs: Any) -> str:
                raise RuntimeError("intentional failure")

        tool = FailingTool()
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "failing_tool"
        tool_call.function.arguments = "{}"

        result = execute_single_native_tool_call(
            tool_call,
            available_functions={"failing_tool": tool._run},
            original_tools=[tool],
            structured_tools=None,
            tools_handler=None,
            agent=None,
            task=None,
            crew=None,
            event_source=MagicMock(),
            printer=None,
            verbose=False,
        )

        assert isinstance(result, NativeToolCallResult)
        assert result.result_as_answer is False
        assert "Error executing tool" in result.result

    def test_result_as_answer_false_when_hook_blocks(self) -> None:
        """When a before-hook blocks a tool with result_as_answer=True, result_as_answer must be False."""
        from unittest.mock import MagicMock

        from crewai.hooks.tool_hooks import (
            clear_before_tool_call_hooks,
            register_before_tool_call_hook,
        )

        class BlockedTool(BaseTool):
            name: str = "blocked_tool"
            description: str = "A tool whose execution will be blocked by a hook"
            result_as_answer: bool = True

            def _run(self, **kwargs: Any) -> str:
                return "should not run"

        tool = BlockedTool()
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "blocked_tool"
        tool_call.function.arguments = "{}"

        register_before_tool_call_hook(lambda _ctx: False)
        try:
            result = execute_single_native_tool_call(
                tool_call,
                available_functions={"blocked_tool": tool._run},
                original_tools=[tool],
                structured_tools=None,
                tools_handler=None,
                agent=None,
                task=None,
                crew=None,
                event_source=MagicMock(),
                printer=None,
                verbose=False,
            )
        finally:
            clear_before_tool_call_hooks()

        assert isinstance(result, NativeToolCallResult)
        assert result.result_as_answer is False
        assert "blocked by hook" in result.result


class TestResolvePlusClient:
    def test_builds_the_default_when_no_client_is_installed(self) -> None:
        from crewai.utilities.agent_utils import resolve_plus_client

        default = MagicMock()

        assert resolve_plus_client(lambda: default) is default

    def test_prefers_an_installed_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A hosted runtime installs a client; the default must not be built,
        since looking up a user credential raises when there isn't one."""
        from crewai.utilities import agent_utils

        installed = MagicMock()
        monkeypatch.setattr(agent_utils, "_create_plus_client_hook", lambda: installed)
        default = MagicMock(side_effect=AssertionError("must not be called"))

        assert agent_utils.resolve_plus_client(default) is installed
        default.assert_not_called()


class TestResolvePlusResponse:
    def test_passes_through_a_sync_response(self) -> None:
        from crewai.utilities.agent_utils import resolve_plus_response

        response = MagicMock()

        assert resolve_plus_response(response) is response

    @pytest.mark.parametrize("inside_loop", [False, True])
    def test_awaits_an_async_response(self, inside_loop: bool) -> None:
        from crewai.utilities.agent_utils import resolve_plus_response

        response = MagicMock()

        async def call() -> Any:
            return response

        if not inside_loop:
            assert resolve_plus_response(call()) is response
            return

        async def main() -> Any:
            return resolve_plus_response(call())

        assert asyncio.run(main()) is response

    def test_carries_context_vars_into_the_worker_thread(self) -> None:
        """Inside a running loop the coroutine runs on another thread; a client
        reading runtime state (the platform token, flow context) must still see
        the caller's values rather than defaults."""
        from crewai.context import get_platform_integration_token, platform_context
        from crewai.utilities.agent_utils import resolve_plus_response

        async def call() -> Any:
            return get_platform_integration_token()

        async def main() -> Any:
            with platform_context("token-from-caller"):
                return resolve_plus_response(call())

        assert asyncio.run(main()) == "token-from-caller"

    def test_rejects_an_awaitable_bound_to_a_loop(self) -> None:
        from crewai.utilities.agent_utils import resolve_plus_response

        async def main() -> None:
            future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
            future.set_result(MagicMock())

            with pytest.raises(TypeError, match="must return a coroutine"):
                resolve_plus_response(future)

        asyncio.run(main())


_FORCE_FINAL_ANSWER = I18N_DEFAULT.errors("force_final_answer")


def _native_tool_history() -> list[dict[str, Any]]:
    """History as the native tool-calling loop leaves it: ends on a user prompt."""
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Collect all the data."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_data", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "name": "get_data", "content": "partial"},
        {"role": "user", "content": I18N_DEFAULT.slice("post_tool_reasoning")},
    ]


def _react_history() -> list[dict[str, Any]]:
    """History as the ReAct loop leaves it: ends on the assistant turn with the observation."""
    return [
        {"role": "system", "content": "You are an agent."},
        {"role": "user", "content": "Collect all the data."},
        {
            "role": "assistant",
            "content": "Thought: I need data\nAction: get_data\nAction Input: {}\nObservation: partial",
        },
    ]


class TestHandleMaxIterationsExceeded:
    """The forced final answer is requested with a user turn, never assistant prefill.

    Current Claude models reject a request whose last message is an assistant
    turn ("This model does not support assistant message prefill"), so the
    nudge must go out as the user's instruction on every loop shape.
    """

    @pytest.mark.parametrize(
        "make_history", [_native_tool_history, _react_history], ids=["native-tools", "react"]
    )
    def test_appends_the_instruction_as_a_user_turn(self, make_history) -> None:
        history = make_history()
        before = [dict(message) for message in history]
        llm = MagicMock()
        llm.call.return_value = "Final Answer: 42"

        result = handle_max_iterations_exceeded(
            printer=MagicMock(), messages=history, llm=llm, callbacks=[], verbose=False
        )

        assert history[:-1] == before
        assert history[-1] == {"role": "user", "content": _FORCE_FINAL_ANSWER}
        llm.call.assert_called_once_with(history, callbacks=[])
        assert isinstance(result, AgentFinish)
        assert result.output == "42"

    def test_action_shaped_reply_still_becomes_a_final_answer(self) -> None:
        reply = "Thought: one more\nAction: get_data\nAction Input: {}"
        llm = MagicMock()
        llm.call.return_value = reply

        result = handle_max_iterations_exceeded(
            printer=MagicMock(), messages=_react_history(), llm=llm, callbacks=[], verbose=False
        )

        assert isinstance(result, AgentFinish)
        assert result.text == reply
        assert result.output == reply

    @pytest.mark.parametrize("reply", [None, ""], ids=["none", "empty"])
    def test_empty_reply_raises(self, reply: str | None) -> None:
        llm = MagicMock()
        llm.call.return_value = reply

        with pytest.raises(ValueError, match="Invalid response from LLM call - None or empty."):
            handle_max_iterations_exceeded(
                printer=MagicMock(), messages=_native_tool_history(), llm=llm, callbacks=[], verbose=False
            )

    @pytest.mark.parametrize("verbose", [True, False])
    def test_notice_is_printed_only_when_verbose(self, verbose: bool) -> None:
        printer = MagicMock()
        llm = MagicMock()
        llm.call.return_value = "Final Answer: 42"

        handle_max_iterations_exceeded(
            printer=printer, messages=_native_tool_history(), llm=llm, callbacks=[], verbose=verbose
        )

        if verbose:
            printer.print.assert_called_once_with(
                content="Maximum iterations reached. Requesting final answer.", color="yellow"
            )
        else:
            printer.print.assert_not_called()

class TestModuleAllowlist:
    """Tests for the tool module allowlist in load_agent_from_repository."""

    def test_is_allowlisted_tool_module_exact_members(self):
        """Exact allowlist members pass; anything else falls through to the package check."""
        from crewai.utilities.agent_utils import _is_allowlisted_tool_module

        assert _is_allowlisted_tool_module("crewai.tools") is True
        assert _is_allowlisted_tool_module("crewai_tools") is True
        assert _is_allowlisted_tool_module("crewai.tools.base_tool") is True
        assert _is_allowlisted_tool_module("os") is False
        assert _is_allowlisted_tool_module("subprocess") is False
        # A loose prefix match must not bless lookalike top-level names.
        assert _is_allowlisted_tool_module("crewai_tools_evil") is False

    def test_crewai_tools_submodule_inside_package_is_allowed(self):
        """A repository record's defining module (e.g. csv_search_tool's own
        submodule) resolves inside the installed crewai_tools package."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from crewai.utilities import agent_utils

        def fake_find_spec(name):
            if name == "crewai_tools":
                return SimpleNamespace(submodule_search_locations=["/site/crewai_tools"])
            if name == "crewai_tools.tools.csv_search_tool.csv_search_tool":
                return SimpleNamespace(
                    origin="/site/crewai_tools/tools/csv_search_tool/csv_search_tool.py"
                )
            raise AssertionError(f"unexpected find_spec({name})")

        with patch.object(
            agent_utils.importlib.util, "find_spec", side_effect=fake_find_spec
        ):
            assert (
                agent_utils._is_allowlisted_tool_module(
                    "crewai_tools.tools.csv_search_tool.csv_search_tool"
                )
                is True
            )

    def test_crewai_tools_submodule_outside_package_is_rejected(self):
        """A dotted name under crewai_tools.* that resolves outside the installed
        package directory is still rejected."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from crewai.utilities import agent_utils

        def fake_find_spec(name):
            if name == "crewai_tools":
                return SimpleNamespace(submodule_search_locations=["/site/crewai_tools"])
            return SimpleNamespace(origin="/tmp/attacker/x.py")

        with patch.object(
            agent_utils.importlib.util, "find_spec", side_effect=fake_find_spec
        ):
            assert (
                agent_utils._is_allowlisted_tool_module("crewai_tools.tools.evil")
                is False
            )

    def test_crewai_tools_submodule_unresolvable_is_rejected(self):
        """A submodule find_spec cannot resolve is rejected."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from crewai.utilities import agent_utils

        def fake_find_spec(name):
            if name == "crewai_tools":
                return SimpleNamespace(submodule_search_locations=["/site/crewai_tools"])
            return None

        with patch.object(
            agent_utils.importlib.util, "find_spec", side_effect=fake_find_spec
        ):
            assert (
                agent_utils._is_allowlisted_tool_module("crewai_tools.tools.missing")
                is False
            )

    def test_crewai_tools_package_missing_is_rejected(self):
        """Without an installed crewai_tools package, submodules are rejected."""
        from unittest.mock import patch

        from crewai.utilities import agent_utils

        with patch.object(
            agent_utils.importlib.util, "find_spec", return_value=None
        ):
            assert (
                agent_utils._is_allowlisted_tool_module(
                    "crewai_tools.tools.csv_search_tool.csv_search_tool"
                )
                is False
            )

    def test_blocked_module_raises_repository_error(self):
        """Loading an agent whose tool references a non-allowlisted module should raise AgentRepositoryError."""
        from unittest.mock import MagicMock, patch

        from crewai.utilities.errors import AgentRepositoryError

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "test-agent",
            "tools": [
                {
                    "module": "os",
                    "name": "system",
                    "init_params": {},
                }
            ],
        }

        with (
            patch("crewai.utilities.agent_utils.resolve_plus_response", return_value=mock_response),
            patch("crewai.utilities.agent_utils.resolve_plus_client"),
        ):
            from crewai.utilities.agent_utils import load_agent_from_repository

            with pytest.raises(AgentRepositoryError, match="not in the allowlist"):
                load_agent_from_repository("test-agent")

    def test_non_basetool_attribute_raises_repository_error(self):
        """A tool name resolving to a non-BaseTool attribute should raise AgentRepositoryError."""
        from unittest.mock import MagicMock, patch

        from crewai.utilities.errors import AgentRepositoryError

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "test-agent",
            "tools": [
                {
                    "module": "crewai.tools",
                    "name": "ToolExecutionFailedError",  # exception class, not a tool
                    "init_params": {},
                }
            ],
        }

        with (
            patch("crewai.utilities.agent_utils.resolve_plus_response", return_value=mock_response),
            patch("crewai.utilities.agent_utils.resolve_plus_client"),
        ):
            from crewai.utilities.agent_utils import load_agent_from_repository

            with pytest.raises(AgentRepositoryError, match="not a BaseTool subclass"):
                load_agent_from_repository("test-agent")

    def test_allowlisted_basetool_instantiation_succeeds(self):
        """An allowlisted module + BaseTool subclass should pass the guard and instantiate."""
        from unittest.mock import MagicMock, patch

        class RepositoryTestTool(BaseTool):
            name: str = "repository_test_tool"
            description: str = "Loads from the repository for testing."

            def _run(self, *args, **kwargs):
                return "ok"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "test-agent",
            "tools": [
                {
                    "module": "crewai.tools",
                    "name": "tool",
                    "init_params": {},
                }
            ],
        }

        with (
            patch("crewai.utilities.agent_utils.resolve_plus_response", return_value=mock_response),
            patch("crewai.utilities.agent_utils.resolve_plus_client"),
            patch("crewai.tools.tool", RepositoryTestTool),
        ):
            from crewai.utilities.agent_utils import load_agent_from_repository

            result = load_agent_from_repository("test-agent")
            assert result.get("name") == "test-agent"

    def test_agent_without_tools_loads_successfully(self):
        """An agent with no tools should load without tool-module validation."""
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "name": "test-agent",
            "tools": [],
        }

        with (
            patch("crewai.utilities.agent_utils.resolve_plus_response", return_value=mock_response),
            patch("crewai.utilities.agent_utils.resolve_plus_client"),
        ):
            from crewai.utilities.agent_utils import load_agent_from_repository

            result = load_agent_from_repository("test-agent")
            assert result.get("name") == "test-agent"


class TestRepositoryToolParamValidation:
    """Tests for init_params validation in load_agent_from_repository.

    The module allowlist gates *which* tool classes may load; init_params
    validation gates *what* a remote record may pass to their constructors.
    """

    @staticmethod
    def _repository_response(tools):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "test-agent", "tools": tools}
        return mock_response

    @staticmethod
    def _load_with_tool(mock_response, tool_class):
        with (
            patch(
                "crewai.utilities.agent_utils.resolve_plus_response",
                return_value=mock_response,
            ),
            patch("crewai.utilities.agent_utils.resolve_plus_client"),
            patch("crewai.tools.tool", tool_class),
        ):
            from crewai.utilities.agent_utils import load_agent_from_repository

            return load_agent_from_repository("test-agent")

    def test_import_string_param_rejected(self):
        """A string init_param for an ImportString field must be rejected:
        pydantic would import the dotted path as a live callable on this host
        (e.g. 'os.system'), which the tool may then invoke."""
        from crewai.utilities.errors import AgentRepositoryError

        class ImportStringTool(BaseTool):
            name: str = "import_string_tool"
            description: str = "Tool with an import-resolving init param."
            custom_embedding_fn: ImportString

            def _run(self, *args, **kwargs):
                return "ok"

        response = self._repository_response(
            [
                {
                    "module": "crewai.tools",
                    "name": "tool",
                    "init_params": {"custom_embedding_fn": "os.system"},
                }
            ]
        )
        with pytest.raises(AgentRepositoryError, match="import-resolving"):
            self._load_with_tool(response, ImportStringTool)

    def test_optional_import_string_param_rejected(self):
        """Optional[ImportString] (and other wrappers) must be caught too."""
        from crewai.utilities.errors import AgentRepositoryError

        class OptionalImportStringTool(BaseTool):
            name: str = "optional_import_string_tool"
            description: str = "Tool with an optional import-resolving param."
            custom_embedding_fn: ImportString | None = None

            def _run(self, *args, **kwargs):
                return "ok"

        response = self._repository_response(
            [
                {
                    "module": "crewai.tools",
                    "name": "tool",
                    "init_params": {"custom_embedding_fn": "builtins.eval"},
                }
            ]
        )
        with pytest.raises(AgentRepositoryError, match="import-resolving"):
            self._load_with_tool(response, OptionalImportStringTool)

    def test_parameterized_import_string_param_rejected(self):
        """ImportString[Callable] (parameterized form) must be caught too.

        The parameterized form is Annotated[Callable, ImportString()] — an
        ImportString *instance* in Annotated metadata, which pydantic may
        keep inside field.annotation or strip into field.metadata depending
        on the version. Either way a remote string like 'os.system' would
        become a live callable on this host.
        """
        from collections.abc import Callable

        from crewai.utilities.errors import AgentRepositoryError

        class ParameterizedImportStringTool(BaseTool):
            name: str = "parameterized_import_string_tool"
            description: str = "Tool with a parameterized import-resolving param."
            custom_embedding_fn: ImportString[Callable[[str], list[float]]] | None = (
                None
            )

            def _run(self, *args, **kwargs):
                return "ok"

        response = self._repository_response(
            [
                {
                    "module": "crewai.tools",
                    "name": "tool",
                    "init_params": {"custom_embedding_fn": "os.system"},
                }
            ]
        )
        with pytest.raises(AgentRepositoryError, match="import-resolving"):
            self._load_with_tool(response, ParameterizedImportStringTool)

    def test_serializable_callable_param_rejected(self):
        """A string init_param for a SerializableCallable field is rejected.

        SerializableCallable's BeforeValidator (string_to_callable) resolves
        a dotted-path string into a live callable at validation time — and
        BaseTool pre-resolves cache_function strings before that. Without
        this gate, a repository record could smuggle 'os.system' into
        cache_function and get arbitrary code execution on this host.
        """
        from crewai.types.callback import SerializableCallable
        from crewai.utilities.errors import AgentRepositoryError

        class SerializableCallableTool(BaseTool):
            name: str = "serializable_callable_tool"
            description: str = "Tool with a SerializableCallable init param."
            cache_function: SerializableCallable | None = None

            def _run(self, *args, **kwargs):
                return "ok"

        response = self._repository_response(
            [
                {
                    "module": "crewai.tools",
                    "name": "tool",
                    "init_params": {"cache_function": "os.system"},
                }
            ]
        )
        with pytest.raises(AgentRepositoryError, match="import-resolving"):
            self._load_with_tool(response, SerializableCallableTool)

    def test_undeclared_param_rejected(self):
        """Records may not smuggle unexpected keyword arguments into the tool
        constructor."""
        from crewai.utilities.errors import AgentRepositoryError

        class ScalarTool(BaseTool):
            name: str = "scalar_tool"
            description: str = "Tool with a plain scalar param."
            top_k: int = 5

            def _run(self, *args, **kwargs):
                return "ok"

        response = self._repository_response(
            [
                {
                    "module": "crewai.tools",
                    "name": "tool",
                    "init_params": {"top_k": 3, "no_such_field": 1},
                }
            ]
        )
        with pytest.raises(AgentRepositoryError, match="does not declare"):
            self._load_with_tool(response, ScalarTool)

    def test_non_dict_init_params_rejected(self):
        """init_params must be a JSON object."""
        from crewai.utilities.errors import AgentRepositoryError

        class ScalarTool(BaseTool):
            name: str = "scalar_tool"
            description: str = "Tool with a plain scalar param."
            top_k: int = 5

            def _run(self, *args, **kwargs):
                return "ok"

        response = self._repository_response(
            [
                {
                    "module": "crewai.tools",
                    "name": "tool",
                    "init_params": ["top_k"],
                }
            ]
        )
        with pytest.raises(AgentRepositoryError, match="expected a JSON object"):
            self._load_with_tool(response, ScalarTool)

    def test_non_json_value_rejected(self):
        """Non-serializable values (notably callables) must never reach the
        tool constructor."""
        from crewai.utilities.agent_utils import _validate_repository_tool_params
        from crewai.utilities.errors import AgentRepositoryError

        class ScalarTool(BaseTool):
            name: str = "scalar_tool"
            description: str = "Tool with a plain scalar param."
            top_k: int = 5

            def _run(self, *args, **kwargs):
                return "ok"

        with pytest.raises(AgentRepositoryError, match="must be JSON-typed"):
            _validate_repository_tool_params(ScalarTool, {"top_k": object()})

    def test_remote_ref_in_args_schema_rejected(self):
        """A remote $ref in a schema dict must not reach the constructor.

        create_model_from_schema resolves $ref with jsonref's default
        loader, so a remote ref would make the loader's host fetch a URL
        (or read a local file via file://) at construction time.
        """
        from crewai.tools import BaseTool
        from crewai.utilities.agent_utils import _validate_repository_tool_params
        from crewai.utilities.errors import AgentRepositoryError

        with pytest.raises(AgentRepositoryError, match=r"non-local \$ref"):
            _validate_repository_tool_params(
                BaseTool,
                {"args_schema": {"$ref": "https://attacker.example/s.json"}},
            )

    def test_nested_remote_ref_in_result_schema_rejected(self):
        """Remote refs are caught even when nested inside the schema."""
        from crewai.tools import BaseTool
        from crewai.utilities.agent_utils import _validate_repository_tool_params
        from crewai.utilities.errors import AgentRepositoryError

        schema = {
            "type": "object",
            "properties": {"payload": {"$ref": "file:///etc/hostname"}},
        }
        with pytest.raises(AgentRepositoryError, match=r"non-local \$ref"):
            _validate_repository_tool_params(BaseTool, {"result_schema": schema})

    def test_no_authority_uri_ref_forms_rejected(self):
        """No-authority URI forms bypass a naive "://"-based check.

        jsonref's default loader treats ``file:/etc/hostname`` and
        ``http:/host/x`` as URIs (a missing authority does not turn them
        into fragments), so both must be rejected.
        """
        from crewai.tools import BaseTool
        from crewai.utilities.agent_utils import _validate_repository_tool_params
        from crewai.utilities.errors import AgentRepositoryError

        for ref in ("file:/etc/hostname", "http:/attacker.example/x.json"):
            schema = {
                "type": "object",
                "properties": {"payload": {"$ref": ref}},
            }
            with pytest.raises(AgentRepositoryError, match=r"non-local \$ref"):
                _validate_repository_tool_params(BaseTool, {"result_schema": schema})

    def test_relative_ref_rejected(self):
        """Relative refs resolve against the process CWD via jsonref's loader.

        ``other.json#/x`` never carries a scheme, so a "://"-based check
        cannot see it — but jsonref resolves it as a local file read.
        """
        from crewai.tools import BaseTool
        from crewai.utilities.agent_utils import _validate_repository_tool_params
        from crewai.utilities.errors import AgentRepositoryError

        schema = {
            "type": "object",
            "properties": {"payload": {"$ref": "other.json#/$defs/x"}},
        }
        with pytest.raises(AgentRepositoryError, match=r"non-local \$ref"):
            _validate_repository_tool_params(BaseTool, {"args_schema": schema})

    def test_plain_schema_dict_still_allowed(self):
        """Schema dicts without remote refs (incl. local fragment refs) pass.

        Legitimate repository records may still customize schemas; only
        network-capable refs are rejected.
        """
        from crewai.tools import BaseTool
        from crewai.utilities.agent_utils import _validate_repository_tool_params

        schema = {
            "type": "object",
            "$defs": {"Name": {"type": "string"}},
            "properties": {"name": {"$ref": "#/$defs/Name"}},
        }
        # Must not raise.
        _validate_repository_tool_params(BaseTool, {"args_schema": schema})

    def test_scalar_params_still_instantiate(self):
        """Legitimate JSON scalar params for declared fields keep working."""
        class ScalarTool(BaseTool):
            name: str = "scalar_tool"
            description: str = "Tool with a plain scalar param."
            top_k: int = 5

            def _run(self, *args, **kwargs):
                return "ok"

        response = self._repository_response(
            [
                {
                    "module": "crewai.tools",
                    "name": "tool",
                    "init_params": {"top_k": 3},
                }
            ]
        )
        result = self._load_with_tool(response, ScalarTool)
        assert result["tools"][0].top_k == 3
