"""Unit tests for AgentExecutor.

Tests the Flow-based agent executor implementation including state management,
flow methods, routing logic, and error handling.
"""

from unittest.mock import Mock, patch

import pytest

from crewai.experimental.agent_executor import (
    AgentReActState,
    AgentExecutor,
)
from crewai.agents.parser import AgentAction, AgentFinish

class TestAgentReActState:
    """Test AgentReActState Pydantic model."""

    def test_state_initialization(self):
        """Test AgentReActState initialization with defaults."""
        state = AgentReActState()
        assert state.iterations == 0
        assert state.messages == []
        assert state.current_answer is None
        assert state.is_finished is False
        assert state.ask_for_human_input is False

    def test_state_with_values(self):
        """Test AgentReActState initialization with values."""
        messages = [{"role": "user", "content": "test"}]
        state = AgentReActState(
            messages=messages,
            iterations=5,
            current_answer=AgentFinish(thought="thinking", output="done", text="final"),
            is_finished=True,
            ask_for_human_input=True,
        )
        assert state.messages == messages
        assert state.iterations == 5
        assert isinstance(state.current_answer, AgentFinish)
        assert state.is_finished is True
        assert state.ask_for_human_input is True


class TestAgentExecutor:
    """Test AgentExecutor class."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for executor."""
        llm = Mock()
        llm.supports_stop_words.return_value = True

        task = Mock()
        task.description = "Test task"
        task.human_input = False
        task.response_model = None

        crew = Mock()
        crew.verbose = False
        crew._train = False

        agent = Mock()
        agent.id = "test-agent-id"
        agent.role = "Test Agent"
        agent.verbose = False
        agent.key = "test-key"

        prompt = {"prompt": "Test prompt with {input}, {tool_names}, {tools}"}

        tools = []
        tools_handler = Mock()

        return {
            "llm": llm,
            "task": task,
            "crew": crew,
            "agent": agent,
            "prompt": prompt,
            "max_iter": 10,
            "tools": tools,
            "tools_names": "",
            "stop_words": ["Observation"],
            "tools_description": "",
            "tools_handler": tools_handler,
        }

    def test_executor_initialization(self, mock_dependencies):
        """Test AgentExecutor initialization."""
        executor = AgentExecutor(**mock_dependencies)

        assert executor.llm == mock_dependencies["llm"]
        assert executor.task == mock_dependencies["task"]
        assert executor.agent == mock_dependencies["agent"]
        assert executor.crew == mock_dependencies["crew"]
        assert executor.max_iter == 10
        assert executor.use_stop_words is True

    def test_initialize_reasoning(self, mock_dependencies):
        """Test flow entry point."""
        with patch.object(
            AgentExecutor, "_show_start_logs"
        ) as mock_show_start:
            executor = AgentExecutor(**mock_dependencies)
            result = executor.initialize_reasoning()

            assert result == "initialized"
            mock_show_start.assert_called_once()

    def test_check_max_iterations_not_reached(self, mock_dependencies):
        """Test routing when iterations < max."""
        executor = AgentExecutor(**mock_dependencies)
        executor.state.iterations = 5

        result = executor.check_max_iterations()
        assert result == "continue_reasoning"

    def test_check_max_iterations_reached(self, mock_dependencies):
        """Test routing when iterations >= max."""
        executor = AgentExecutor(**mock_dependencies)
        executor.state.iterations = 10

        result = executor.check_max_iterations()
        assert result == "force_final_answer"

    def test_route_by_answer_type_action(self, mock_dependencies):
        """Test routing for AgentAction."""
        executor = AgentExecutor(**mock_dependencies)
        executor.state.current_answer = AgentAction(
            thought="thinking", tool="search", tool_input="query", text="action text"
        )

        result = executor.route_by_answer_type()
        assert result == "execute_tool"

    def test_route_by_answer_type_finish(self, mock_dependencies):
        """Test routing for AgentFinish."""
        executor = AgentExecutor(**mock_dependencies)
        executor.state.current_answer = AgentFinish(
            thought="final thoughts", output="Final answer", text="complete"
        )

        result = executor.route_by_answer_type()
        assert result == "agent_finished"

    def test_continue_iteration(self, mock_dependencies):
        """Test iteration continuation."""
        executor = AgentExecutor(**mock_dependencies)

        result = executor.continue_iteration()

        assert result == "check_iteration"

    def test_finalize_success(self, mock_dependencies):
        """Test finalize with valid AgentFinish."""
        with patch.object(AgentExecutor, "_show_logs") as mock_show_logs:
            executor = AgentExecutor(**mock_dependencies)
            executor.state.current_answer = AgentFinish(
                thought="final thinking", output="Done", text="complete"
            )

            result = executor.finalize()

            assert result == "completed"
            assert executor.state.is_finished is True
            mock_show_logs.assert_called_once()

    def test_finalize_failure(self, mock_dependencies):
        """Test finalize skips when given AgentAction instead of AgentFinish."""
        executor = AgentExecutor(**mock_dependencies)
        executor.state.current_answer = AgentAction(
            thought="thinking", tool="search", tool_input="query", text="action text"
        )

        result = executor.finalize()

        # Should return "skipped" and not set is_finished
        assert result == "skipped"
        assert executor.state.is_finished is False

    def test_format_prompt(self, mock_dependencies):
        """Test prompt formatting."""
        executor = AgentExecutor(**mock_dependencies)
        inputs = {"input": "test input", "tool_names": "tool1, tool2", "tools": "desc"}

        result = executor._format_prompt("Prompt {input} {tool_names} {tools}", inputs)

        assert "test input" in result
        assert "tool1, tool2" in result
        assert "desc" in result

    def test_is_training_mode_false(self, mock_dependencies):
        """Test training mode detection when not in training."""
        executor = AgentExecutor(**mock_dependencies)
        assert executor._is_training_mode() is False

    def test_is_training_mode_true(self, mock_dependencies):
        """Test training mode detection when in training."""
        mock_dependencies["crew"]._train = True
        executor = AgentExecutor(**mock_dependencies)
        assert executor._is_training_mode() is True

    def test_append_message_to_state(self, mock_dependencies):
        """Test message appending to state."""
        executor = AgentExecutor(**mock_dependencies)
        initial_count = len(executor.state.messages)

        executor._append_message_to_state("test message")

        assert len(executor.state.messages) == initial_count + 1
        assert executor.state.messages[-1]["content"] == "test message"

    def test_invoke_step_callback(self, mock_dependencies):
        """Test step callback invocation."""
        callback = Mock()
        mock_dependencies["step_callback"] = callback

        executor = AgentExecutor(**mock_dependencies)
        answer = AgentFinish(thought="thinking", output="test", text="final")

        executor._invoke_step_callback(answer)

        callback.assert_called_once_with(answer)

    def test_invoke_step_callback_none(self, mock_dependencies):
        """Test step callback when none provided."""
        mock_dependencies["step_callback"] = None
        executor = AgentExecutor(**mock_dependencies)

        # Should not raise error
        executor._invoke_step_callback(
            AgentFinish(thought="thinking", output="test", text="final")
        )

    @patch("crewai.experimental.agent_executor.handle_output_parser_exception")
    def test_recover_from_parser_error(
        self, mock_handle_exception, mock_dependencies
    ):
        """Test recovery from OutputParserError."""
        from crewai.agents.parser import OutputParserError

        mock_handle_exception.return_value = None

        executor = AgentExecutor(**mock_dependencies)
        executor._last_parser_error = OutputParserError("test error")
        initial_iterations = executor.state.iterations

        result = executor.recover_from_parser_error()

        assert result == "initialized"
        assert executor.state.iterations == initial_iterations + 1
        mock_handle_exception.assert_called_once()

    @patch("crewai.experimental.agent_executor.handle_context_length")
    def test_recover_from_context_length(
        self, mock_handle_context, mock_dependencies
    ):
        """Test recovery from context length error."""
        executor = AgentExecutor(**mock_dependencies)
        executor._last_context_error = Exception("context too long")
        initial_iterations = executor.state.iterations

        result = executor.recover_from_context_length()

        assert result == "initialized"
        assert executor.state.iterations == initial_iterations + 1
        mock_handle_context.assert_called_once()

    def test_use_stop_words_property(self, mock_dependencies):
        """Test use_stop_words property."""
        mock_dependencies["llm"].supports_stop_words.return_value = True
        executor = AgentExecutor(**mock_dependencies)
        assert executor.use_stop_words is True

        mock_dependencies["llm"].supports_stop_words.return_value = False
        executor = AgentExecutor(**mock_dependencies)
        assert executor.use_stop_words is False

    def test_compatibility_properties(self, mock_dependencies):
        """Test compatibility properties for mixin."""
        executor = AgentExecutor(**mock_dependencies)
        executor.state.messages = [{"role": "user", "content": "test"}]
        executor.state.iterations = 5

        # Test that compatibility properties return state values
        assert executor.messages == executor.state.messages
        assert executor.iterations == executor.state.iterations


class TestFlowErrorHandling:
    """Test error handling in flow methods."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        llm = Mock()
        llm.supports_stop_words.return_value = True

        task = Mock()
        task.description = "Test task"

        crew = Mock()
        agent = Mock()
        agent.role = "Test Agent"
        agent.verbose = False

        prompt = {"prompt": "Test {input}"}

        return {
            "llm": llm,
            "task": task,
            "crew": crew,
            "agent": agent,
            "prompt": prompt,
            "max_iter": 10,
            "tools": [],
            "tools_names": "",
            "stop_words": [],
            "tools_description": "",
            "tools_handler": Mock(),
        }

    @patch("crewai.experimental.agent_executor.get_llm_response")
    @patch("crewai.experimental.agent_executor.enforce_rpm_limit")
    def test_call_llm_parser_error(
        self, mock_enforce_rpm, mock_get_llm, mock_dependencies
    ):
        """Test call_llm_and_parse handles OutputParserError."""
        from crewai.agents.parser import OutputParserError

        mock_enforce_rpm.return_value = None
        mock_get_llm.side_effect = OutputParserError("parse failed")

        executor = AgentExecutor(**mock_dependencies)
        result = executor.call_llm_and_parse()

        assert result == "parser_error"
        assert executor._last_parser_error is not None

    @patch("crewai.experimental.agent_executor.get_llm_response")
    @patch("crewai.experimental.agent_executor.enforce_rpm_limit")
    @patch("crewai.experimental.agent_executor.is_context_length_exceeded")
    def test_call_llm_context_error(
        self,
        mock_is_context_exceeded,
        mock_enforce_rpm,
        mock_get_llm,
        mock_dependencies,
    ):
        """Test call_llm_and_parse handles context length error."""
        mock_enforce_rpm.return_value = None
        mock_get_llm.side_effect = Exception("context length")
        mock_is_context_exceeded.return_value = True

        executor = AgentExecutor(**mock_dependencies)
        result = executor.call_llm_and_parse()

        assert result == "context_error"
        assert executor._last_context_error is not None


class TestFlowInvoke:
    """Test the invoke method that maintains backward compatibility."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        llm = Mock()
        task = Mock()
        task.description = "Test"
        task.human_input = False

        crew = Mock()
        crew._short_term_memory = None
        crew._long_term_memory = None
        crew._entity_memory = None
        crew._external_memory = None

        agent = Mock()
        agent.role = "Test"
        agent.verbose = False

        prompt = {"prompt": "Test {input} {tool_names} {tools}"}

        return {
            "llm": llm,
            "task": task,
            "crew": crew,
            "agent": agent,
            "prompt": prompt,
            "max_iter": 10,
            "tools": [],
            "tools_names": "",
            "stop_words": [],
            "tools_description": "",
            "tools_handler": Mock(),
        }

    @patch.object(AgentExecutor, "kickoff")
    @patch.object(AgentExecutor, "_create_short_term_memory")
    @patch.object(AgentExecutor, "_create_long_term_memory")
    @patch.object(AgentExecutor, "_create_external_memory")
    def test_invoke_success(
        self,
        mock_external_memory,
        mock_long_term_memory,
        mock_short_term_memory,
        mock_kickoff,
        mock_dependencies,
    ):
        """Test successful invoke without human feedback."""
        executor = AgentExecutor(**mock_dependencies)

        # Mock kickoff to set the final answer in state
        def mock_kickoff_side_effect():
            executor.state.current_answer = AgentFinish(
                thought="final thinking", output="Final result", text="complete"
            )

        mock_kickoff.side_effect = mock_kickoff_side_effect

        inputs = {"input": "test", "tool_names": "", "tools": ""}
        result = executor.invoke(inputs)

        assert result == {"output": "Final result"}
        mock_kickoff.assert_called_once()
        mock_short_term_memory.assert_called_once()
        mock_long_term_memory.assert_called_once()
        mock_external_memory.assert_called_once()

    @patch.object(AgentExecutor, "kickoff")
    def test_invoke_failure_no_agent_finish(self, mock_kickoff, mock_dependencies):
        """Test invoke fails without AgentFinish."""
        executor = AgentExecutor(**mock_dependencies)
        executor.state.current_answer = AgentAction(
            thought="thinking", tool="test", tool_input="test", text="action text"
        )

        inputs = {"input": "test", "tool_names": "", "tools": ""}

        with pytest.raises(RuntimeError, match="without reaching a final answer"):
            executor.invoke(inputs)

    @patch.object(AgentExecutor, "kickoff")
    @patch.object(AgentExecutor, "_create_short_term_memory")
    @patch.object(AgentExecutor, "_create_long_term_memory")
    @patch.object(AgentExecutor, "_create_external_memory")
    def test_invoke_with_system_prompt(
        self,
        mock_external_memory,
        mock_long_term_memory,
        mock_short_term_memory,
        mock_kickoff,
        mock_dependencies,
    ):
        """Test invoke with system prompt configuration."""
        mock_dependencies["prompt"] = {
            "system": "System: {input}",
            "user": "User: {input} {tool_names} {tools}",
        }
        executor = AgentExecutor(**mock_dependencies)

        def mock_kickoff_side_effect():
            executor.state.current_answer = AgentFinish(
                thought="final thoughts", output="Done", text="complete"
            )

        mock_kickoff.side_effect = mock_kickoff_side_effect

        inputs = {"input": "test", "tool_names": "", "tools": ""}
        result = executor.invoke(inputs)
        mock_short_term_memory.assert_called_once()
        mock_long_term_memory.assert_called_once()
        mock_external_memory.assert_called_once()
        mock_kickoff.assert_called_once()

        assert result == {"output": "Done"}
        assert len(executor.state.messages) >= 2
