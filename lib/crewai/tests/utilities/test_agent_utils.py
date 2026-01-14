"""Unit tests for agent_utils module.

Tests the utility functions for agent execution including tool extraction
and LLM response handling.
"""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from pydantic import BaseModel, Field

from crewai.tools.structured_tool import CrewStructuredTool
from crewai.utilities.agent_utils import (
    _extract_tools_from_context,
    aget_llm_response,
    get_llm_response,
)
from crewai.utilities.printer import Printer


class MockArgsSchema(BaseModel):
    """Mock args schema for testing."""

    query: str = Field(description="The search query")


class TestExtractToolsFromContext:
    """Test _extract_tools_from_context function."""

    def test_returns_none_when_context_is_none(self):
        """Test that None is returned when executor_context is None."""
        result = _extract_tools_from_context(None)
        assert result is None

    def test_returns_none_when_no_tools_attribute(self):
        """Test that None is returned when context has no tools."""
        mock_context = Mock(spec=[])
        result = _extract_tools_from_context(mock_context)
        assert result is None

    def test_returns_none_when_tools_is_empty(self):
        """Test that None is returned when tools list is empty."""
        mock_context = Mock()
        mock_context.tools = []
        result = _extract_tools_from_context(mock_context)
        assert result is None

    def test_extracts_tools_from_crew_agent_executor(self):
        """Test tool extraction from CrewAgentExecutor (has 'tools' attribute)."""
        mock_tool = CrewStructuredTool(
            name="search_tool",
            description="A tool for searching",
            args_schema=MockArgsSchema,
            func=lambda query: f"Results for {query}",
        )

        mock_context = Mock()
        mock_context.tools = [mock_tool]

        result = _extract_tools_from_context(mock_context)

        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "search_tool"
        assert result[0]["description"] == "A tool for searching"
        assert result[0]["args_schema"] == MockArgsSchema

    def test_extracts_tools_from_lite_agent(self):
        """Test tool extraction from LiteAgent (has '_parsed_tools' attribute)."""
        mock_tool = CrewStructuredTool(
            name="calculator_tool",
            description="A tool for calculations",
            args_schema=MockArgsSchema,
            func=lambda query: f"Calculated {query}",
        )

        mock_context = Mock(spec=["_parsed_tools"])
        mock_context._parsed_tools = [mock_tool]

        result = _extract_tools_from_context(mock_context)

        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "calculator_tool"
        assert result[0]["description"] == "A tool for calculations"
        assert result[0]["args_schema"] == MockArgsSchema

    def test_extracts_multiple_tools(self):
        """Test extraction of multiple tools."""
        tool1 = CrewStructuredTool(
            name="tool1",
            description="First tool",
            args_schema=MockArgsSchema,
            func=lambda query: "result1",
        )
        tool2 = CrewStructuredTool(
            name="tool2",
            description="Second tool",
            args_schema=MockArgsSchema,
            func=lambda query: "result2",
        )

        mock_context = Mock()
        mock_context.tools = [tool1, tool2]

        result = _extract_tools_from_context(mock_context)

        assert result is not None
        assert len(result) == 2
        assert result[0]["name"] == "tool1"
        assert result[1]["name"] == "tool2"

    def test_prefers_tools_over_parsed_tools(self):
        """Test that 'tools' attribute is preferred over '_parsed_tools'."""
        tool_from_tools = CrewStructuredTool(
            name="from_tools",
            description="Tool from tools attribute",
            args_schema=MockArgsSchema,
            func=lambda query: "from_tools",
        )
        tool_from_parsed = CrewStructuredTool(
            name="from_parsed",
            description="Tool from _parsed_tools attribute",
            args_schema=MockArgsSchema,
            func=lambda query: "from_parsed",
        )

        mock_context = Mock()
        mock_context.tools = [tool_from_tools]
        mock_context._parsed_tools = [tool_from_parsed]

        result = _extract_tools_from_context(mock_context)

        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "from_tools"


class TestGetLlmResponse:
    """Test get_llm_response function."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM."""
        llm = Mock()
        llm.call = Mock(return_value="LLM response")
        return llm

    @pytest.fixture
    def mock_printer(self):
        """Create a mock printer."""
        return Mock(spec=Printer)

    def test_passes_tools_to_llm_call(self, mock_llm, mock_printer):
        """Test that tools are extracted and passed to llm.call()."""
        mock_tool = CrewStructuredTool(
            name="test_tool",
            description="A test tool",
            args_schema=MockArgsSchema,
            func=lambda query: "result",
        )

        mock_context = Mock()
        mock_context.tools = [mock_tool]
        mock_context.messages = [{"role": "user", "content": "test"}]
        mock_context.before_llm_call_hooks = []
        mock_context.after_llm_call_hooks = []

        with patch(
            "crewai.utilities.agent_utils._setup_before_llm_call_hooks",
            return_value=True,
        ):
            with patch(
                "crewai.utilities.agent_utils._setup_after_llm_call_hooks",
                return_value="LLM response",
            ):
                result = get_llm_response(
                    llm=mock_llm,
                    messages=[{"role": "user", "content": "test"}],
                    callbacks=[],
                    printer=mock_printer,
                    executor_context=mock_context,
                )

        # Verify llm.call was called with tools parameter
        mock_llm.call.assert_called_once()
        call_kwargs = mock_llm.call.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "test_tool"

    def test_passes_none_tools_when_no_context(self, mock_llm, mock_printer):
        """Test that tools=None is passed when no executor_context."""
        result = get_llm_response(
            llm=mock_llm,
            messages=[{"role": "user", "content": "test"}],
            callbacks=[],
            printer=mock_printer,
            executor_context=None,
        )

        mock_llm.call.assert_called_once()
        call_kwargs = mock_llm.call.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is None

    def test_passes_none_tools_when_context_has_no_tools(
        self, mock_llm, mock_printer
    ):
        """Test that tools=None is passed when context has no tools."""
        mock_context = Mock()
        mock_context.tools = []
        mock_context.messages = [{"role": "user", "content": "test"}]
        mock_context.before_llm_call_hooks = []
        mock_context.after_llm_call_hooks = []

        with patch(
            "crewai.utilities.agent_utils._setup_before_llm_call_hooks",
            return_value=True,
        ):
            with patch(
                "crewai.utilities.agent_utils._setup_after_llm_call_hooks",
                return_value="LLM response",
            ):
                result = get_llm_response(
                    llm=mock_llm,
                    messages=[{"role": "user", "content": "test"}],
                    callbacks=[],
                    printer=mock_printer,
                    executor_context=mock_context,
                )

        mock_llm.call.assert_called_once()
        call_kwargs = mock_llm.call.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is None


class TestAgetLlmResponse:
    """Test aget_llm_response async function."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM with async call."""
        llm = Mock()
        llm.acall = AsyncMock(return_value="Async LLM response")
        return llm

    @pytest.fixture
    def mock_printer(self):
        """Create a mock printer."""
        return Mock(spec=Printer)

    @pytest.mark.asyncio
    async def test_passes_tools_to_llm_acall(self, mock_llm, mock_printer):
        """Test that tools are extracted and passed to llm.acall()."""
        mock_tool = CrewStructuredTool(
            name="async_test_tool",
            description="An async test tool",
            args_schema=MockArgsSchema,
            func=lambda query: "async result",
        )

        mock_context = Mock()
        mock_context.tools = [mock_tool]
        mock_context.messages = [{"role": "user", "content": "async test"}]
        mock_context.before_llm_call_hooks = []
        mock_context.after_llm_call_hooks = []

        with patch(
            "crewai.utilities.agent_utils._setup_before_llm_call_hooks",
            return_value=True,
        ):
            with patch(
                "crewai.utilities.agent_utils._setup_after_llm_call_hooks",
                return_value="Async LLM response",
            ):
                result = await aget_llm_response(
                    llm=mock_llm,
                    messages=[{"role": "user", "content": "async test"}],
                    callbacks=[],
                    printer=mock_printer,
                    executor_context=mock_context,
                )

        # Verify llm.acall was called with tools parameter
        mock_llm.acall.assert_called_once()
        call_kwargs = mock_llm.acall.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["name"] == "async_test_tool"

    @pytest.mark.asyncio
    async def test_passes_none_tools_when_no_context(self, mock_llm, mock_printer):
        """Test that tools=None is passed when no executor_context."""
        result = await aget_llm_response(
            llm=mock_llm,
            messages=[{"role": "user", "content": "test"}],
            callbacks=[],
            printer=mock_printer,
            executor_context=None,
        )

        mock_llm.acall.assert_called_once()
        call_kwargs = mock_llm.acall.call_args[1]
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is None


class TestToolsPassedToGeminiModels:
    """Test that tools are properly passed for Gemini models.

    This test class specifically addresses GitHub issue #4238 where
    Gemini models fail with UNEXPECTED_TOOL_CALL errors because tools
    were not being passed to llm.call().
    """

    @pytest.fixture
    def mock_gemini_llm(self):
        """Create a mock Gemini LLM."""
        llm = Mock()
        llm.model = "gemini/gemini-2.0-flash-exp"
        llm.call = Mock(return_value="Gemini response with tool call")
        return llm

    @pytest.fixture
    def mock_printer(self):
        """Create a mock printer."""
        return Mock(spec=Printer)

    @pytest.fixture
    def delegation_tools(self):
        """Create mock delegation tools similar to hierarchical crew setup."""

        class DelegateWorkArgsSchema(BaseModel):
            task: str = Field(description="The task to delegate")
            context: str = Field(description="Context for the task")
            coworker: str = Field(description="The coworker to delegate to")

        class AskQuestionArgsSchema(BaseModel):
            question: str = Field(description="The question to ask")
            context: str = Field(description="Context for the question")
            coworker: str = Field(description="The coworker to ask")

        delegate_tool = CrewStructuredTool(
            name="Delegate work to coworker",
            description="Delegate a specific task to one of your coworkers",
            args_schema=DelegateWorkArgsSchema,
            func=lambda task, context, coworker: f"Delegated {task} to {coworker}",
        )

        ask_question_tool = CrewStructuredTool(
            name="Ask question to coworker",
            description="Ask a specific question to one of your coworkers",
            args_schema=AskQuestionArgsSchema,
            func=lambda question, context, coworker: f"Asked {question} to {coworker}",
        )

        return [delegate_tool, ask_question_tool]

    def test_gemini_receives_tools_for_hierarchical_crew(
        self, mock_gemini_llm, mock_printer, delegation_tools
    ):
        """Test that Gemini models receive tools when used in hierarchical crew.

        This test verifies the fix for issue #4238 where the manager agent
        in a hierarchical crew would fail because tools weren't passed to
        the Gemini model, causing UNEXPECTED_TOOL_CALL errors.
        """
        mock_context = Mock()
        mock_context.tools = delegation_tools
        mock_context.messages = [
            {"role": "system", "content": "You are a manager agent"},
            {"role": "user", "content": "Coordinate the team to answer this question"},
        ]
        mock_context.before_llm_call_hooks = []
        mock_context.after_llm_call_hooks = []

        with patch(
            "crewai.utilities.agent_utils._setup_before_llm_call_hooks",
            return_value=True,
        ):
            with patch(
                "crewai.utilities.agent_utils._setup_after_llm_call_hooks",
                return_value="Gemini response with tool call",
            ):
                result = get_llm_response(
                    llm=mock_gemini_llm,
                    messages=mock_context.messages,
                    callbacks=[],
                    printer=mock_printer,
                    executor_context=mock_context,
                )

        # Verify that tools were passed to the Gemini model
        mock_gemini_llm.call.assert_called_once()
        call_kwargs = mock_gemini_llm.call.call_args[1]

        assert "tools" in call_kwargs
        assert call_kwargs["tools"] is not None
        assert len(call_kwargs["tools"]) == 2

        # Verify the delegation tools are properly formatted
        tool_names = [t["name"] for t in call_kwargs["tools"]]
        assert "Delegate work to coworker" in tool_names
        assert "Ask question to coworker" in tool_names

        # Verify each tool has the required fields
        for tool_dict in call_kwargs["tools"]:
            assert "name" in tool_dict
            assert "description" in tool_dict
            assert "args_schema" in tool_dict

    def test_tool_dict_format_compatible_with_llm_providers(
        self, mock_gemini_llm, mock_printer, delegation_tools
    ):
        """Test that extracted tools are in a format compatible with LLM providers.

        The tool dictionaries should have 'name', 'description', and 'args_schema'
        fields that can be processed by the LLM's _prepare_completion_params method.
        """
        mock_context = Mock()
        mock_context.tools = delegation_tools
        mock_context.messages = [{"role": "user", "content": "test"}]
        mock_context.before_llm_call_hooks = []
        mock_context.after_llm_call_hooks = []

        with patch(
            "crewai.utilities.agent_utils._setup_before_llm_call_hooks",
            return_value=True,
        ):
            with patch(
                "crewai.utilities.agent_utils._setup_after_llm_call_hooks",
                return_value="response",
            ):
                get_llm_response(
                    llm=mock_gemini_llm,
                    messages=mock_context.messages,
                    callbacks=[],
                    printer=mock_printer,
                    executor_context=mock_context,
                )

        call_kwargs = mock_gemini_llm.call.call_args[1]
        tools = call_kwargs["tools"]

        for tool_dict in tools:
            # Verify the format matches what extract_tool_info() in common.py expects
            assert isinstance(tool_dict["name"], str)
            assert isinstance(tool_dict["description"], str)
            # args_schema should be a Pydantic model class
            assert hasattr(tool_dict["args_schema"], "model_json_schema")
