"""Tests for human approval functionality in hooks."""

from __future__ import annotations

from unittest.mock import Mock, patch

from crewai.hooks.llm_hooks import LLMCallHookContext
from crewai.hooks.tool_hooks import ToolCallHookContext
import pytest


@pytest.fixture
def mock_executor():
    """Create a mock executor for LLM hook context."""
    executor = Mock()
    executor.messages = [{"role": "system", "content": "Test message"}]
    executor.agent = Mock(role="Test Agent")
    executor.task = Mock(description="Test Task")
    executor.crew = Mock()
    executor.llm = Mock()
    executor.iterations = 0
    return executor


@pytest.fixture
def mock_tool():
    """Create a mock tool for tool hook context."""
    tool = Mock()
    tool.name = "test_tool"
    tool.description = "Test tool description"
    return tool


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = Mock()
    agent.role = "Test Agent"
    return agent


@pytest.fixture
def mock_task():
    """Create a mock task."""
    task = Mock()
    task.description = "Test task"
    return task


class TestLLMHookHumanInput:
    """Test request_human_input() on LLMCallHookContext."""

    @patch("builtins.input", return_value="test response")
    @patch("crewai.hooks.llm_hooks.event_listener")
    def test_request_human_input_returns_user_response(
        self, mock_event_listener, mock_input, mock_executor
    ):
        """Test that request_human_input returns the user's input."""
        # Setup mock formatter
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        context = LLMCallHookContext(executor=mock_executor)

        response = context.request_human_input(
            prompt="Test prompt", default_message="Test default message"
        )

        assert response == "test response"
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="")
    @patch("crewai.hooks.llm_hooks.event_listener")
    def test_request_human_input_returns_empty_string_on_enter(
        self, mock_event_listener, mock_input, mock_executor
    ):
        """Test that pressing Enter returns empty string."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        context = LLMCallHookContext(executor=mock_executor)

        response = context.request_human_input(prompt="Test")

        assert response == ""
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="test")
    @patch("crewai.hooks.llm_hooks.event_listener")
    def test_request_human_input_pauses_and_resumes_live_updates(
        self, mock_event_listener, mock_input, mock_executor
    ):
        """Test that live updates are paused and resumed."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        context = LLMCallHookContext(executor=mock_executor)

        context.request_human_input(prompt="Test")

        # Verify pause was called
        mock_formatter.pause_live_updates.assert_called_once()

        # Verify resume was called
        mock_formatter.resume_live_updates.assert_called_once()

    @patch("builtins.input", side_effect=Exception("Input error"))
    @patch("crewai.hooks.llm_hooks.event_listener")
    def test_request_human_input_resumes_on_exception(
        self, mock_event_listener, mock_input, mock_executor
    ):
        """Test that live updates are resumed even if input raises exception."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        context = LLMCallHookContext(executor=mock_executor)

        with pytest.raises(Exception, match="Input error"):
            context.request_human_input(prompt="Test")

        # Verify resume was still called (in finally block)
        mock_formatter.resume_live_updates.assert_called_once()

    @patch("builtins.input", return_value="  test response  ")
    @patch("crewai.hooks.llm_hooks.event_listener")
    def test_request_human_input_strips_whitespace(
        self, mock_event_listener, mock_input, mock_executor
    ):
        """Test that user input is stripped of leading/trailing whitespace."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        context = LLMCallHookContext(executor=mock_executor)

        response = context.request_human_input(prompt="Test")

        assert response == "test response"  # Whitespace stripped


class TestToolHookHumanInput:
    """Test request_human_input() on ToolCallHookContext."""

    @patch("builtins.input", return_value="approve")
    @patch("crewai.hooks.tool_hooks.event_listener")
    def test_request_human_input_returns_user_response(
        self, mock_event_listener, mock_input, mock_tool, mock_agent, mock_task
    ):
        """Test that request_human_input returns the user's input."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input={"arg": "value"},
            tool=mock_tool,
            agent=mock_agent,
            task=mock_task,
        )

        response = context.request_human_input(
            prompt="Approve this tool?", default_message="Type 'approve':"
        )

        assert response == "approve"
        mock_input.assert_called_once()

    @patch("builtins.input", return_value="")
    @patch("crewai.hooks.tool_hooks.event_listener")
    def test_request_human_input_handles_empty_input(
        self, mock_event_listener, mock_input, mock_tool
    ):
        """Test that empty input (Enter key) is handled correctly."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input={},
            tool=mock_tool,
        )

        response = context.request_human_input(prompt="Test")

        assert response == ""

    @patch("builtins.input", return_value="test")
    @patch("crewai.hooks.tool_hooks.event_listener")
    def test_request_human_input_pauses_and_resumes(
        self, mock_event_listener, mock_input, mock_tool
    ):
        """Test that live updates are properly paused and resumed."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input={},
            tool=mock_tool,
        )

        context.request_human_input(prompt="Test")

        mock_formatter.pause_live_updates.assert_called_once()
        mock_formatter.resume_live_updates.assert_called_once()

    @patch("builtins.input", side_effect=KeyboardInterrupt)
    @patch("crewai.hooks.tool_hooks.event_listener")
    def test_request_human_input_resumes_on_keyboard_interrupt(
        self, mock_event_listener, mock_input, mock_tool
    ):
        """Test that live updates are resumed even on keyboard interrupt."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input={},
            tool=mock_tool,
        )

        with pytest.raises(KeyboardInterrupt):
            context.request_human_input(prompt="Test")

        # Verify resume was still called (in finally block)
        mock_formatter.resume_live_updates.assert_called_once()


class TestApprovalHookIntegration:
    """Test integration scenarios with approval hooks."""

    @patch("builtins.input", return_value="approve")
    @patch("crewai.hooks.tool_hooks.event_listener")
    def test_approval_hook_allows_execution(
        self, mock_event_listener, mock_input, mock_tool
    ):
        """Test that approval hook allows execution when approved."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        def approval_hook(context: ToolCallHookContext) -> bool | None:
            response = context.request_human_input(
                prompt="Approve?", default_message="Type 'approve':"
            )
            return None if response == "approve" else False

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input={},
            tool=mock_tool,
        )

        result = approval_hook(context)

        assert result is None  # Allowed
        assert mock_input.called

    @patch("builtins.input", return_value="deny")
    @patch("crewai.hooks.tool_hooks.event_listener")
    def test_approval_hook_blocks_execution(
        self, mock_event_listener, mock_input, mock_tool
    ):
        """Test that approval hook blocks execution when denied."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        def approval_hook(context: ToolCallHookContext) -> bool | None:
            response = context.request_human_input(
                prompt="Approve?", default_message="Type 'approve':"
            )
            return None if response == "approve" else False

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input={},
            tool=mock_tool,
        )

        result = approval_hook(context)

        assert result is False  # Blocked
        assert mock_input.called

    @patch("builtins.input", return_value="modified result")
    @patch("crewai.hooks.tool_hooks.event_listener")
    def test_review_hook_modifies_result(
        self, mock_event_listener, mock_input, mock_tool
    ):
        """Test that review hook can modify tool results."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        def review_hook(context: ToolCallHookContext) -> str | None:
            response = context.request_human_input(
                prompt="Review result",
                default_message="Press Enter to keep, or provide modified version:",
            )
            return response if response else None

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input={},
            tool=mock_tool,
            tool_result="original result",
        )

        modified_result = review_hook(context)

        assert modified_result == "modified result"
        assert mock_input.called

    @patch("builtins.input", return_value="")
    @patch("crewai.hooks.tool_hooks.event_listener")
    def test_review_hook_keeps_original_on_enter(
        self, mock_event_listener, mock_input, mock_tool
    ):
        """Test that pressing Enter keeps original result."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        def review_hook(context: ToolCallHookContext) -> str | None:
            response = context.request_human_input(
                prompt="Review result", default_message="Press Enter to keep:"
            )
            return response if response else None

        context = ToolCallHookContext(
            tool_name="test_tool",
            tool_input={},
            tool=mock_tool,
            tool_result="original result",
        )

        modified_result = review_hook(context)

        assert modified_result is None  # Keep original


class TestCostControlApproval:
    """Test cost control approval hook scenarios."""

    @patch("builtins.input", return_value="yes")
    @patch("crewai.hooks.llm_hooks.event_listener")
    def test_cost_control_allows_when_approved(
        self, mock_event_listener, mock_input, mock_executor
    ):
        """Test that expensive calls are allowed when approved."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        # Set high iteration count
        mock_executor.iterations = 10

        def cost_control_hook(context: LLMCallHookContext) -> None:
            if context.iterations > 5:
                response = context.request_human_input(
                    prompt=f"Iteration {context.iterations} - expensive call",
                    default_message="Type 'yes' to continue:",
                )
                if response.lower() != "yes":
                    print("Call blocked")

        context = LLMCallHookContext(executor=mock_executor)

        # Should not raise exception and should call input
        cost_control_hook(context)
        assert mock_input.called

    @patch("builtins.input", return_value="no")
    @patch("crewai.hooks.llm_hooks.event_listener")
    def test_cost_control_logs_when_denied(
        self, mock_event_listener, mock_input, mock_executor
    ):
        """Test that denied calls are logged."""
        mock_formatter = Mock()
        mock_event_listener.formatter = mock_formatter

        mock_executor.iterations = 10

        messages_logged = []

        def cost_control_hook(context: LLMCallHookContext) -> None:
            if context.iterations > 5:
                response = context.request_human_input(
                    prompt=f"Iteration {context.iterations}",
                    default_message="Type 'yes' to continue:",
                )
                if response.lower() != "yes":
                    messages_logged.append("blocked")

        context = LLMCallHookContext(executor=mock_executor)

        cost_control_hook(context)

        assert len(messages_logged) == 1
        assert messages_logged[0] == "blocked"
