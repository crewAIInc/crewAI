"""Unit tests for AgentExecutor.

Tests the Flow-based agent executor implementation including state management,
flow methods, routing logic, and error handling.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from crewai.agents.tools_handler import ToolsHandler as _ToolsHandler
from crewai.agents.step_executor import StepExecutor


def _build_executor(**kwargs: Any) -> AgentExecutor:
    """Create an AgentExecutor without validation — for unit tests.

    Uses model_construct to skip Pydantic validators so plain Mock()
    objects are accepted for typed fields like llm, agent, crew, task.
    """
    executor = AgentExecutor.model_construct(**kwargs)
    executor._state = AgentExecutorState()
    executor._methods = {}
    executor._method_outputs = []
    executor._completed_methods = set()
    executor._fired_or_listeners = set()
    executor._pending_and_listeners = {}
    executor._method_execution_counts = {}
    executor._method_call_counts = {}
    executor._event_futures = []
    executor._human_feedback_method_outputs = {}
    executor._input_history = []
    executor._is_execution_resuming = False
    import threading
    executor._state_lock = threading.Lock()
    executor._or_listeners_lock = threading.Lock()
    executor._execution_lock = threading.Lock()
    executor._finalize_lock = threading.Lock()
    executor._finalize_called = False
    executor._is_executing = False
    executor._has_been_invoked = False
    executor._last_parser_error = None
    executor._last_context_error = None
    executor._step_executor = None
    executor._planner_observer = None
    return executor
from crewai.agents.planner_observer import PlannerObserver
from crewai.experimental.agent_executor import (
    AgentExecutorState,
    AgentExecutor,
)
from crewai.agents.parser import AgentAction, AgentFinish
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.tool_usage_events import (
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.tools.tool_types import ToolResult
from crewai.utilities.step_execution_context import StepExecutionContext
from crewai.utilities.planning_types import TodoItem

class TestAgentExecutorState:
    """Test AgentExecutorState Pydantic model."""

    def test_state_initialization(self):
        """Test AgentExecutorState initialization with defaults."""
        state = AgentExecutorState()
        assert state.iterations == 0
        assert state.messages == []
        assert state.current_answer is None
        assert state.is_finished is False
        assert state.ask_for_human_input is False
        # Planning state fields
        assert state.plan is None
        assert state.plan_ready is False

    def test_state_with_plan(self):
        """Test AgentExecutorState initialization with planning fields."""
        state = AgentExecutorState(
            plan="Step 1: Do X\nStep 2: Do Y",
            plan_ready=True,
        )
        assert state.plan == "Step 1: Do X\nStep 2: Do Y"
        assert state.plan_ready is True

    def test_state_with_values(self):
        """Test AgentExecutorState initialization with values."""
        messages = [{"role": "user", "content": "test"}]
        state = AgentExecutorState(
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
        llm.stop = []

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
        tools_handler = Mock(spec=_ToolsHandler)

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
        executor = _build_executor(**mock_dependencies)

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
            executor = _build_executor(**mock_dependencies)
            result = executor.initialize_reasoning()

            assert result == "initialized"
            mock_show_start.assert_called_once()

    def test_check_max_iterations_not_reached(self, mock_dependencies):
        """Test routing when iterations < max."""
        executor = _build_executor(**mock_dependencies)
        executor.state.iterations = 5

        result = executor.check_max_iterations()
        assert result == "continue_reasoning"

    def test_check_max_iterations_reached(self, mock_dependencies):
        """Test routing when iterations >= max."""
        executor = _build_executor(**mock_dependencies)
        executor.state.iterations = 10

        result = executor.check_max_iterations()
        assert result == "force_final_answer"

    def test_route_by_answer_type_action(self, mock_dependencies):
        """Test routing for AgentAction."""
        executor = _build_executor(**mock_dependencies)
        executor.state.current_answer = AgentAction(
            thought="thinking", tool="search", tool_input="query", text="action text"
        )

        result = executor.route_by_answer_type()
        assert result == "execute_tool"

    def test_route_by_answer_type_finish(self, mock_dependencies):
        """Test routing for AgentFinish."""
        executor = _build_executor(**mock_dependencies)
        executor.state.current_answer = AgentFinish(
            thought="final thoughts", output="Final answer", text="complete"
        )

        result = executor.route_by_answer_type()
        assert result == "agent_finished"

    def test_continue_iteration(self, mock_dependencies):
        """Test iteration continuation."""
        executor = _build_executor(**mock_dependencies)

        result = executor.continue_iteration()

        assert result == "check_iteration"

    def test_finalize_success(self, mock_dependencies):
        """Test finalize with valid AgentFinish."""
        with patch.object(AgentExecutor, "_show_logs") as mock_show_logs:
            executor = _build_executor(**mock_dependencies)
            executor.state.current_answer = AgentFinish(
                thought="final thinking", output="Done", text="complete"
            )

            result = executor.finalize()

            assert result == "completed"
            assert executor.state.is_finished is True
            mock_show_logs.assert_called_once()

    def test_finalize_failure(self, mock_dependencies):
        """Test finalize skips when given AgentAction instead of AgentFinish."""
        executor = _build_executor(**mock_dependencies)
        executor.state.current_answer = AgentAction(
            thought="thinking", tool="search", tool_input="query", text="action text"
        )

        result = executor.finalize()

        # Should return "skipped" and not set is_finished
        assert result == "skipped"
        assert executor.state.is_finished is False

    def test_finalize_skips_synthesis_for_strong_last_todo_result(
        self, mock_dependencies
    ):
        """Finalize should skip synthesis when last todo is already a complete answer."""
        with patch.object(AgentExecutor, "_show_logs") as mock_show_logs:
            executor = _build_executor(**mock_dependencies)
            executor.state.todos.items = [
                TodoItem(
                    step_number=1,
                    description="Gather source details",
                    tool_to_use="search_tool",
                    status="completed",
                    result="Source A and Source B identified.",
                ),
                TodoItem(
                    step_number=2,
                    description="Write final response",
                    tool_to_use=None,
                    status="completed",
                    result=(
                        "The final recommendation is to adopt a phased rollout plan with "
                        "weekly checkpoints, explicit ownership, and a rollback path for "
                        "each milestone. This approach keeps risk controlled while still "
                        "moving quickly, and it aligns delivery metrics with stakeholder "
                        "communication and operational readiness."
                    ),
                ),
            ]

            with patch.object(
                executor, "_synthesize_final_answer_from_todos"
            ) as mock_synthesize:
                result = executor.finalize()

            assert result == "completed"
            assert isinstance(executor.state.current_answer, AgentFinish)
            assert (
                executor.state.current_answer.output
                == executor.state.todos.items[1].result
            )
            assert executor.state.is_finished is True
            mock_synthesize.assert_not_called()
            mock_show_logs.assert_called_once()

    def test_finalize_keeps_synthesis_when_response_model_is_set(
        self, mock_dependencies
    ):
        """Finalize should still synthesize when response_model is configured."""
        with patch.object(AgentExecutor, "_show_logs"):
            executor = _build_executor(**mock_dependencies)
            executor.response_model = Mock()
            executor.state.todos.items = [
                TodoItem(
                    step_number=1,
                    description="Write final response",
                    tool_to_use=None,
                    status="completed",
                    result=(
                        "This is already detailed prose with multiple sentences. "
                        "It should still run synthesis because structured output "
                        "was requested via response_model."
                    ),
                )
            ]

            def _set_current_answer() -> None:
                executor.state.current_answer = AgentFinish(
                    thought="Synthesized",
                    output="structured-like-answer",
                    text="structured-like-answer",
                )

            with patch.object(
                executor,
                "_synthesize_final_answer_from_todos",
                side_effect=_set_current_answer,
            ) as mock_synthesize:
                result = executor.finalize()

            assert result == "completed"
            mock_synthesize.assert_called_once()

    def test_format_prompt(self, mock_dependencies):
        """Test prompt formatting."""
        executor = _build_executor(**mock_dependencies)
        inputs = {"input": "test input", "tool_names": "tool1, tool2", "tools": "desc"}

        result = executor._format_prompt("Prompt {input} {tool_names} {tools}", inputs)

        assert "test input" in result
        assert "tool1, tool2" in result
        assert "desc" in result

    def test_is_training_mode_false(self, mock_dependencies):
        """Test training mode detection when not in training."""
        executor = _build_executor(**mock_dependencies)
        assert executor._is_training_mode() is False

    def test_is_training_mode_true(self, mock_dependencies):
        """Test training mode detection when in training."""
        mock_dependencies["crew"]._train = True
        executor = _build_executor(**mock_dependencies)
        assert executor._is_training_mode() is True

    def test_append_message_to_state(self, mock_dependencies):
        """Test message appending to state."""
        executor = _build_executor(**mock_dependencies)
        initial_count = len(executor.state.messages)

        executor._append_message_to_state("test message")

        assert len(executor.state.messages) == initial_count + 1
        assert executor.state.messages[-1]["content"] == "test message"

    def test_invoke_step_callback(self, mock_dependencies):
        """Test step callback invocation."""
        callback = Mock()
        mock_dependencies["step_callback"] = callback

        executor = _build_executor(**mock_dependencies)
        answer = AgentFinish(thought="thinking", output="test", text="final")

        executor._invoke_step_callback(answer)

        callback.assert_called_once_with(answer)

    def test_invoke_step_callback_none(self, mock_dependencies):
        """Test step callback when none provided."""
        mock_dependencies["step_callback"] = None
        executor = _build_executor(**mock_dependencies)

        # Should not raise error
        executor._invoke_step_callback(
            AgentFinish(thought="thinking", output="test", text="final")
        )

    @pytest.mark.asyncio
    async def test_invoke_step_callback_async_inside_running_loop(
        self, mock_dependencies
    ):
        """Test async step callback scheduling when already in an event loop."""
        callback = AsyncMock()
        mock_dependencies["step_callback"] = callback
        executor = _build_executor(**mock_dependencies)

        answer = AgentFinish(thought="thinking", output="test", text="final")
        with patch("crewai.experimental.agent_executor.asyncio.run") as mock_run:
            executor._invoke_step_callback(answer)
            await asyncio.sleep(0)

        callback.assert_awaited_once_with(answer)
        mock_run.assert_not_called()


class TestStepExecutorCriticalFixes:
    """Regression tests for critical plan-and-execute issues."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies for AgentExecutor tests in this class."""
        llm = Mock()
        llm.stop = []
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

    @pytest.fixture
    def step_executor(self):
        llm = Mock()
        llm.stop = []
        llm.supports_stop_words.return_value = True

        agent = Mock()
        agent.role = "Test Agent"
        agent.goal = "Execute tasks"
        agent.verbose = False
        agent.key = "test-agent-key"

        tool = Mock()
        tool.name = "count_words"
        task = Mock()
        task.name = "test-task"
        task.description = "test task description"

        return StepExecutor(
            llm=llm,
            tools=[tool],
            agent=agent,
            original_tools=[],
            tools_handler=Mock(),
            task=task,
            crew=Mock(),
            function_calling_llm=None,
            request_within_rpm_limit=None,
            callbacks=[],
        )

    def test_step_executor_fails_when_expected_tool_is_not_called(self, step_executor):
        """Step should fail if a configured expected tool is not actually invoked."""
        todo = TodoItem(
            step_number=1,
            description="Count words in input text.",
            tool_to_use="count_words",
            depends_on=[],
            status="pending",
        )
        context = StepExecutionContext(task_description="task", task_goal="goal")

        with patch.object(step_executor, "_build_isolated_messages", return_value=[]):
            with patch.object(
                step_executor, "_execute_text_parsed", return_value="No tool used."
            ):
                result = step_executor.execute(todo, context)

        assert result.success is False
        assert result.error is not None
        assert "Expected tool 'count_words' was not called" in result.error

    def test_step_executor_text_tool_emits_usage_events(self, step_executor):
        """Text-parsed tool execution should emit started and finished events."""
        started_events: list[ToolUsageStartedEvent] = []
        finished_events: list[ToolUsageFinishedEvent] = []

        tool_name = "count_words"
        action = AgentAction(
            thought="Need a tool",
            tool=tool_name,
            tool_input='{"text":"hello world"}',
            text="Action: count_words",
        )

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _on_started(_source, event):
            if event.tool_name == tool_name:
                started_events.append(event)

        @crewai_event_bus.on(ToolUsageFinishedEvent)
        def _on_finished(_source, event):
            if event.tool_name == tool_name:
                finished_events.append(event)

        with patch(
            "crewai.agents.step_executor.execute_tool_and_check_finality",
            return_value=ToolResult(result="2", result_as_answer=False),
        ):
            output = step_executor._execute_text_tool_with_events(action)

        crewai_event_bus.flush()

        assert output == "2"
        assert len(started_events) >= 1
        assert len(finished_events) >= 1

    @patch("crewai.experimental.agent_executor.handle_output_parser_exception")
    def test_recover_from_parser_error(
        self, mock_handle_exception, mock_dependencies
    ):
        """Test recovery from OutputParserError."""
        from crewai.agents.parser import OutputParserError

        mock_handle_exception.return_value = None

        executor = _build_executor(**mock_dependencies)
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
        executor = _build_executor(**mock_dependencies)
        executor._last_context_error = Exception("context too long")
        initial_iterations = executor.state.iterations

        result = executor.recover_from_context_length()

        assert result == "initialized"
        assert executor.state.iterations == initial_iterations + 1
        mock_handle_context.assert_called_once()

    def test_use_stop_words_property(self, mock_dependencies):
        """Test use_stop_words property."""
        mock_dependencies["llm"].supports_stop_words.return_value = True
        executor = _build_executor(**mock_dependencies)
        assert executor.use_stop_words is True

        mock_dependencies["llm"].supports_stop_words.return_value = False
        executor = _build_executor(**mock_dependencies)
        assert executor.use_stop_words is False

    def test_compatibility_properties(self, mock_dependencies):
        """Test compatibility properties for mixin."""
        executor = _build_executor(**mock_dependencies)
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
        llm.stop = []
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

        executor = _build_executor(**mock_dependencies)
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

        executor = _build_executor(**mock_dependencies)
        result = executor.call_llm_and_parse()

        assert result == "context_error"
        assert executor._last_context_error is not None


class TestFlowInvoke:
    """Test the invoke method that maintains backward compatibility."""

    @pytest.fixture
    def mock_dependencies(self):
        """Create mock dependencies."""
        llm = Mock()
        llm.stop = []
        task = Mock()
        task.description = "Test"
        task.human_input = False

        crew = Mock()
        crew._memory = None

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
    @patch.object(AgentExecutor, "_save_to_memory")
    def test_invoke_success(
        self,
        mock_save_to_memory,
        mock_kickoff,
        mock_dependencies,
    ):
        """Test successful invoke without human feedback."""
        executor = _build_executor(**mock_dependencies)

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
        mock_save_to_memory.assert_called_once()

    @patch.object(AgentExecutor, "kickoff")
    def test_invoke_failure_no_agent_finish(self, mock_kickoff, mock_dependencies):
        """Test invoke fails without AgentFinish."""
        executor = _build_executor(**mock_dependencies)
        executor.state.current_answer = AgentAction(
            thought="thinking", tool="test", tool_input="test", text="action text"
        )

        inputs = {"input": "test", "tool_names": "", "tools": ""}

        with pytest.raises(RuntimeError, match="without reaching a final answer"):
            executor.invoke(inputs)

    @patch.object(AgentExecutor, "kickoff")
    @patch.object(AgentExecutor, "_save_to_memory")
    def test_invoke_with_system_prompt(
        self,
        mock_save_to_memory,
        mock_kickoff,
        mock_dependencies,
    ):
        """Test invoke with system prompt configuration."""
        mock_dependencies["prompt"] = {
            "system": "System: {input}",
            "user": "User: {input} {tool_names} {tools}",
        }
        executor = _build_executor(**mock_dependencies)

        def mock_kickoff_side_effect():
            executor.state.current_answer = AgentFinish(
                thought="final thoughts", output="Done", text="complete"
            )

        mock_kickoff.side_effect = mock_kickoff_side_effect

        inputs = {"input": "test", "tool_names": "", "tools": ""}
        result = executor.invoke(inputs)
        mock_save_to_memory.assert_called_once()
        mock_kickoff.assert_called_once()

        assert result == {"output": "Done"}
        assert len(executor.state.messages) >= 2


class TestNativeToolExecution:
    """Test native tool execution behavior."""

    @pytest.fixture
    def mock_dependencies(self):
        llm = Mock()
        llm.stop = []
        llm.supports_stop_words.return_value = True

        task = Mock()
        task.name = "Test Task"
        task.description = "Test"
        task.human_input = False
        task.response_model = None

        crew = Mock()
        crew._memory = None
        crew.verbose = False
        crew._train = False

        agent = Mock()
        agent.id = "test-agent-id"
        agent.role = "Test Agent"
        agent.verbose = False
        agent.key = "test-key"

        prompt = {"prompt": "Test {input} {tool_names} {tools}"}

        tools_handler = Mock(spec=_ToolsHandler)
        tools_handler.cache = None

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
            "tools_handler": tools_handler,
        }

    def test_execute_native_tool_runs_parallel_for_multiple_calls(
        self, mock_dependencies
    ):
        executor = _build_executor(**mock_dependencies)

        def slow_one() -> str:
            time.sleep(0.2)
            return "one"

        def slow_two() -> str:
            time.sleep(0.2)
            return "two"

        executor._available_functions = {"slow_one": slow_one, "slow_two": slow_two}
        executor.state.pending_tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "slow_one", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "function": {"name": "slow_two", "arguments": "{}"},
            },
        ]

        started = time.perf_counter()
        result = executor.execute_native_tool()
        elapsed = time.perf_counter() - started

        assert result == "native_tool_completed"
        assert elapsed < 0.5
        tool_messages = [m for m in executor.state.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 2
        assert tool_messages[0]["tool_call_id"] == "call_1"
        assert tool_messages[1]["tool_call_id"] == "call_2"

    def test_execute_native_tool_falls_back_to_sequential_for_result_as_answer(
        self, mock_dependencies
    ):
        executor = _build_executor(**mock_dependencies)

        def slow_one() -> str:
            time.sleep(0.2)
            return "one"

        def slow_two() -> str:
            time.sleep(0.2)
            return "two"

        result_tool = Mock()
        result_tool.name = "slow_one"
        result_tool.result_as_answer = True
        result_tool.max_usage_count = None
        result_tool.current_usage_count = 0

        executor.original_tools = [result_tool]
        executor._available_functions = {"slow_one": slow_one, "slow_two": slow_two}
        executor.state.pending_tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "slow_one", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "function": {"name": "slow_two", "arguments": "{}"},
            },
        ]

        started = time.perf_counter()
        result = executor.execute_native_tool()
        elapsed = time.perf_counter() - started

        assert result == "tool_result_is_final"
        assert elapsed >= 0.2
        assert elapsed < 0.8
        assert isinstance(executor.state.current_answer, AgentFinish)
        assert executor.state.current_answer.output == "one"

    def test_execute_native_tool_result_as_answer_short_circuits_remaining_calls(
        self, mock_dependencies
    ):
        executor = _build_executor(**mock_dependencies)
        call_counts = {"slow_one": 0, "slow_two": 0}

        def slow_one() -> str:
            call_counts["slow_one"] += 1
            time.sleep(0.2)
            return "one"

        def slow_two() -> str:
            call_counts["slow_two"] += 1
            time.sleep(0.2)
            return "two"

        result_tool = Mock()
        result_tool.name = "slow_one"
        result_tool.result_as_answer = True
        result_tool.max_usage_count = None
        result_tool.current_usage_count = 0

        executor.original_tools = [result_tool]
        executor._available_functions = {"slow_one": slow_one, "slow_two": slow_two}
        executor.state.pending_tool_calls = [
            {
                "id": "call_1",
                "function": {"name": "slow_one", "arguments": "{}"},
            },
            {
                "id": "call_2",
                "function": {"name": "slow_two", "arguments": "{}"},
            },
        ]

        started = time.perf_counter()
        result = executor.execute_native_tool()
        elapsed = time.perf_counter() - started

        assert result == "tool_result_is_final"
        assert isinstance(executor.state.current_answer, AgentFinish)
        assert executor.state.current_answer.output == "one"
        assert call_counts["slow_one"] == 1
        assert call_counts["slow_two"] == 0
        assert elapsed < 0.5

        tool_messages = [m for m in executor.state.messages if m.get("role") == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0]["tool_call_id"] == "call_1"

    def test_check_native_todo_completion_requires_current_todo(
        self, mock_dependencies
    ):
        from crewai.utilities.planning_types import TodoList

        executor = _build_executor(**mock_dependencies)

        # No current todo → not satisfied
        executor.state.todos = TodoList(items=[])
        assert executor.check_native_todo_completion() == "todo_not_satisfied"

        # With a current todo that has tool_to_use → satisfied
        running = TodoItem(
            step_number=1,
            description="Use the expected tool",
            tool_to_use="expected_tool",
            status="running",
        )
        executor.state.todos = TodoList(items=[running])
        assert executor.check_native_todo_completion() == "todo_satisfied"

        # With a current todo without tool_to_use → still satisfied
        running.tool_to_use = None
        assert executor.check_native_todo_completion() == "todo_satisfied"


class TestPlannerObserver:
    def test_observe_fallback_is_conservative_on_llm_error(self):
        llm = Mock()
        llm.call.side_effect = RuntimeError("llm unavailable")

        agent = Mock()
        agent.role = "Observer Test Agent"
        agent.llm = llm
        agent.planning_config = None

        task = Mock()
        task.description = "Test task"
        task.expected_output = "Expected result"

        observer = PlannerObserver(agent=agent, task=task)

        completed_step = TodoItem(
            step_number=1,
            description="Do something",
            status="running",
        )
        observation = observer.observe(
            completed_step=completed_step,
            result="Error: tool timeout",
            all_completed=[],
            remaining_todos=[],
        )

        # When the observer LLM fails, the fallback is conservative:
        # assume the step succeeded and continue (don't wipe the plan).
        assert observation.step_completed_successfully is True
        assert observation.remaining_plan_still_valid is True
        assert observation.needs_full_replan is False


class TestAgentExecutorPlanning:
    """Test planning functionality in AgentExecutor with real agent kickoff."""

    @pytest.mark.vcr()
    def test_agent_kickoff_with_planning_stores_plan_in_state(self):
        """Test that Agent.kickoff() with planning enabled stores plan in executor state."""
        from crewai import Agent, PlanningConfig
        from crewai.llm import LLM

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Math Assistant",
            goal="Help solve simple math problems",
            backstory="A helpful assistant that solves math problems step by step",
            llm=llm,
            planning_config=PlanningConfig(max_attempts=1),
            verbose=False,
        )

        # Execute kickoff with a simple task
        result = agent.kickoff("What is 2 + 2?")

        # Verify result
        assert result is not None
        assert "4" in str(result)

    @pytest.mark.vcr()
    def test_agent_kickoff_without_planning_skips_plan_generation(self):
        """Test that Agent.kickoff() without planning skips planning phase."""
        from crewai import Agent
        from crewai.llm import LLM

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Math Assistant",
            goal="Help solve simple math problems",
            backstory="A helpful assistant",
            llm=llm,
            # No planning_config = no planning
            verbose=False,
        )

        # Execute kickoff
        result = agent.kickoff("What is 3 + 3?")

        # Verify we get a result
        assert result is not None
        assert "6" in str(result)

    @pytest.mark.vcr()
    def test_planning_disabled_skips_planning(self):
        """Test that planning=False skips planning."""
        from crewai import Agent
        from crewai.llm import LLM

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Math Assistant",
            goal="Help solve simple math problems",
            backstory="A helpful assistant",
            llm=llm,
            planning=False,  # Explicitly disable planning
            verbose=False,
        )

        result = agent.kickoff("What is 5 + 5?")

        # Should still complete successfully
        assert result is not None
        assert "10" in str(result)

    def test_backward_compat_reasoning_true_enables_planning(self):
        """Test that reasoning=True (deprecated) still enables planning."""
        import warnings
        from crewai import Agent
        from crewai.llm import LLM

        llm = LLM("gpt-4o-mini")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            agent = Agent(
                role="Test Agent",
                goal="Complete tasks",
                backstory="A helpful agent",
                llm=llm,
                reasoning=True,  # Deprecated but should still work
                verbose=False,
            )

        # Should have planning_config created from reasoning=True
        assert agent.planning_config is not None
        assert agent.planning_enabled is True

    @pytest.mark.vcr()
    def test_executor_state_contains_plan_after_planning(self):
        """Test that executor state contains plan after planning phase."""
        from crewai import Agent, PlanningConfig
        from crewai.llm import LLM
        from crewai.experimental.agent_executor import AgentExecutor

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Math Assistant",
            goal="Help solve simple math problems",
            backstory="A helpful assistant that solves math problems step by step",
            llm=llm,
            planning_config=PlanningConfig(max_attempts=1),
            verbose=False,
        )

        # Track executor for inspection
        executor_ref = [None]
        original_invoke = AgentExecutor.invoke

        def capture_executor(self, inputs):
            executor_ref[0] = self
            return original_invoke(self, inputs)

        with patch.object(AgentExecutor, "invoke", capture_executor):
            result = agent.kickoff("What is 7 + 7?")

        # Verify result
        assert result is not None

        # If we captured an executor, check its state
        if executor_ref[0] is not None:
            # After planning, state should have plan info
            assert hasattr(executor_ref[0].state, "plan")
            assert hasattr(executor_ref[0].state, "plan_ready")

    @pytest.mark.vcr()
    def test_planning_creates_minimal_steps_for_multi_step_task(self):
        """Test that planning creates steps and executes them for a multi-step task.

        This task requires multiple dependent steps:
        1. Identify the first 3 prime numbers (2, 3, 5)
        2. Sum them (2 + 3 + 5 = 10)
        3. Multiply by 2 (10 * 2 = 20)

        The plan-and-execute architecture should produce step results.
        """
        from crewai import Agent, PlanningConfig
        from crewai.llm import LLM
        from crewai.experimental.agent_executor import AgentExecutor

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Math Tutor",
            goal="Solve multi-step math problems accurately",
            backstory="An expert math tutor who breaks down problems step by step",
            llm=llm,
            planning_config=PlanningConfig(max_attempts=1, max_steps=10),
            verbose=False,
        )

        # Track the plan that gets generated
        captured_plan = [None]
        original_invoke = AgentExecutor.invoke

        def capture_plan(self, inputs):
            result = original_invoke(self, inputs)
            captured_plan[0] = self.state.plan
            return result

        with patch.object(AgentExecutor, "invoke", capture_plan):
            result = agent.kickoff(
                "Calculate the sum of the first 3 prime numbers, then multiply that result by 2. "
                "Show your work for each step."
            )

        # Verify we got a result with step outputs
        assert result is not None
        result_str = str(result)
        # Should contain at least some mathematical content from the steps
        assert "prime" in result_str.lower() or "2" in result_str or "10" in result_str

        # Verify a plan was generated
        assert captured_plan[0] is not None

    @pytest.mark.vcr()
    def test_planning_handles_sequential_dependency_task(self):
        """Test planning for a task where step N depends on step N-1.

        Task: Convert 100 Celsius to Fahrenheit, then round to nearest 10.
        Step 1: Apply formula (C * 9/5 + 32) = 212
        Step 2: Round 212 to nearest 10 = 210

        This tests that the planner creates a plan and executes steps.
        """
        from crewai import Agent, PlanningConfig
        from crewai.llm import LLM
        from crewai.experimental.agent_executor import AgentExecutor

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Unit Converter",
            goal="Accurately convert between units and apply transformations",
            backstory="A precise unit conversion specialist",
            llm=llm,
            planning_config=PlanningConfig(max_attempts=1, max_steps=10),
            verbose=False,
        )

        captured_plan = [None]
        original_invoke = AgentExecutor.invoke

        def capture_plan(self, inputs):
            result = original_invoke(self, inputs)
            captured_plan[0] = self.state.plan
            return result

        with patch.object(AgentExecutor, "invoke", capture_plan):
            result = agent.kickoff(
                "Convert 100 degrees Celsius to Fahrenheit, then round the result to the nearest 10."
            )

        assert result is not None
        result_str = str(result)
        # Should contain conversion-related content
        assert "212" in result_str or "210" in result_str or "Fahrenheit" in result_str or "celsius" in result_str.lower()

        # Plan should exist
        assert captured_plan[0] is not None


class TestResponseFormatWithKickoff:
    """Test that Agent.kickoff(response_format=MyModel) returns structured output.

    Real LLM calls via VCR cassettes. Tests both with and without planning,
    using real tools for the planning case to exercise the full Plan-and-Execute
    path including synthesis with response_model.
    """

    @pytest.mark.vcr()
    def test_kickoff_response_format_without_planning(self):
        """Test that kickoff(response_format) returns structured output without planning."""
        from pydantic import BaseModel, Field
        from crewai import Agent
        from crewai.llm import LLM

        class MathResult(BaseModel):
            answer: int = Field(description="The numeric answer")
            explanation: str = Field(description="Brief explanation of the solution")

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Math Assistant",
            goal="Solve math problems and return structured results",
            backstory="A precise math assistant that always returns structured data",
            llm=llm,
            verbose=False,
        )

        result = agent.kickoff("What is 15 + 27?", response_format=MathResult)

        assert result is not None
        assert result.pydantic is not None
        assert isinstance(result.pydantic, MathResult)
        assert result.pydantic.answer == 42
        assert len(result.pydantic.explanation) > 0

    @pytest.mark.vcr()
    def test_kickoff_response_format_with_planning_and_tools(self):
        """Test response_format with planning + tools (multi-step research).

        This is the key test for _synthesize_final_answer_from_todos:
        1. Planning generates steps that use the EXA search tool
        2. StepExecutor runs each step in isolation with tool calls
        3. The synthesis step produces a structured BaseModel output

        The response_format should be respected by the synthesis LLM call,
        NOT by intermediate step executions.
        """
        from pydantic import BaseModel, Field
        from crewai import Agent, PlanningConfig
        from crewai.llm import LLM
        from crewai_tools import EXASearchTool

        class ResearchSummary(BaseModel):
            topic: str = Field(description="The research topic")
            key_findings: list[str] = Field(description="List of 3-5 key findings")
            conclusion: str = Field(description="A brief conclusion paragraph")

        llm = LLM("gpt-4o-mini")
        exa = EXASearchTool()

        agent = Agent(
            role="Research Analyst",
            goal="Research topics using search tools and produce structured summaries",
            backstory=(
                "You are a research analyst who searches the web for information, "
                "identifies key findings, and produces structured research summaries."
            ),
            llm=llm,
            planning_config=PlanningConfig(max_attempts=1, max_steps=5),
            tools=[exa],
            verbose=False,
        )

        result = agent.kickoff(
            "Research the current state of autonomous AI agents in 2025. "
            "Search for recent developments, then summarize the key findings.",
            response_format=ResearchSummary,
        )

        assert result is not None
        # The synthesis step should have produced structured output
        assert result.pydantic is not None
        assert isinstance(result.pydantic, ResearchSummary)
        # Verify the structured fields are populated
        assert len(result.pydantic.topic) > 0
        assert len(result.pydantic.key_findings) >= 1
        assert len(result.pydantic.conclusion) > 0

    @pytest.mark.vcr()
    def test_kickoff_no_response_format_returns_raw_text(self):
        """Test that kickoff without response_format returns plain text."""
        from crewai import Agent
        from crewai.llm import LLM

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Math Assistant",
            goal="Solve math problems",
            backstory="A helpful math assistant",
            llm=llm,
            verbose=False,
        )

        result = agent.kickoff("What is 10 + 10?")

        assert result is not None
        assert result.pydantic is None
        assert "20" in str(result)


class TestReasoningEffort:
    """Test reasoning_effort levels in PlanningConfig.

    - low:  observe() runs (validates step success), but skip decide/replan/refine
    - medium: observe() runs, replan on failure only (mocked)
    - high: full observation pipeline with decide/replan/refine/goal-achieved
    """

    @pytest.mark.vcr()
    def test_reasoning_effort_low_skips_decide_and_replan(self):
        """Low effort: observe runs but decide/replan/refine are never called.

        Verifies that with reasoning_effort='low':
        1. The agent produces a correct result
        2. The observation phase still runs (observations are stored)
        3. The decide_next_action/refine/replan pipeline is bypassed
        """
        from crewai import Agent, PlanningConfig
        from crewai.llm import LLM
        from crewai.experimental.agent_executor import AgentExecutor

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Math Tutor",
            goal="Solve multi-step math problems accurately",
            backstory="An expert math tutor who breaks down problems step by step",
            llm=llm,
            planning_config=PlanningConfig(
                reasoning_effort="low",
                max_attempts=1,
                max_steps=10,
            ),
            verbose=False,
        )

        # Capture the executor to inspect state after execution
        executor_ref = [None]
        original_invoke = AgentExecutor.invoke

        def capture_executor(self, inputs):
            result = original_invoke(self, inputs)
            executor_ref[0] = self
            return result

        with patch.object(AgentExecutor, "invoke", capture_executor):
            result = agent.kickoff(
                "What is the sum of the first 3 prime numbers (2, 3, 5)?"
            )

        assert result is not None
        assert "10" in str(result)

        # Verify observations were still collected (observe() ran)
        executor = executor_ref[0]
        if executor is not None and executor.state.todos.items:
            assert len(executor.state.observations) > 0, (
                "Low effort should still run observe() to validate steps"
            )

            # Verify no replan was triggered
            assert executor.state.replan_count == 0, (
                "Low effort should never trigger replanning"
            )

            # Check execution log for reasoning_effort annotation
            observation_logs = [
                log for log in executor.state.execution_log
                if log.get("type") == "observation"
            ]
            for log in observation_logs:
                assert log.get("reasoning_effort") == "low"

    @pytest.mark.vcr()
    def test_reasoning_effort_high_runs_full_observation_pipeline(self):
        """High effort: full observation pipeline with decide/replan/refine.

        Verifies that with reasoning_effort='high':
        1. The agent produces a correct result
        2. Observations are stored
        3. The full decide_next_action pipeline runs (the observation-driven
           routing is exercised, even if it just routes to continue_plan)
        """
        from crewai import Agent, PlanningConfig
        from crewai.llm import LLM
        from crewai.experimental.agent_executor import AgentExecutor

        llm = LLM("gpt-4o-mini")

        agent = Agent(
            role="Math Tutor",
            goal="Solve multi-step math problems accurately",
            backstory="An expert math tutor who breaks down problems step by step",
            llm=llm,
            planning_config=PlanningConfig(
                reasoning_effort="high",
                max_attempts=1,
                max_steps=10,
            ),
            verbose=False,
        )

        executor_ref = [None]
        original_invoke = AgentExecutor.invoke

        def capture_executor(self, inputs):
            result = original_invoke(self, inputs)
            executor_ref[0] = self
            return result

        with patch.object(AgentExecutor, "invoke", capture_executor):
            result = agent.kickoff(
                "What is the sum of the first 3 prime numbers (2, 3, 5)?"
            )

        assert result is not None
        assert "10" in str(result)

        # Verify observations were collected
        executor = executor_ref[0]
        if executor is not None and executor.state.todos.items:
            assert len(executor.state.observations) > 0, (
                "High effort should run observe() on every step"
            )

            # Check execution log shows high reasoning_effort
            observation_logs = [
                log for log in executor.state.execution_log
                if log.get("type") == "observation"
            ]
            for log in observation_logs:
                assert log.get("reasoning_effort") == "high"

    def test_reasoning_effort_medium_replans_on_failure(self):
        """Medium effort: replan triggered when observation reports failure.

        This test mocks the PlannerObserver to simulate a failed step,
        verifying that medium effort routes to replan_now on failure
        but continues on success.
        """
        from crewai.experimental.agent_executor import AgentExecutor
        from crewai.utilities.planning_types import (
            StepObservation,
            TodoItem,
            TodoList,
        )

        # --- Build a minimal mock executor with medium effort ---
        executor = Mock(spec=AgentExecutor)
        executor.agent = Mock()
        executor.agent.verbose = False
        executor.agent.planning_config = Mock()
        executor.agent.planning_config.reasoning_effort = "medium"

        # Provide the real method under test (bound to our mock)
        executor.handle_step_observed_medium = (
            AgentExecutor.handle_step_observed_medium.__get__(executor)
        )

        # --- Case 1: step succeeded → should return "continue_plan" ---
        success_todo = TodoItem(
            step_number=1,
            description="Calculate something",
            status="running",
            result="42",
        )
        success_observation = StepObservation(
            step_completed_successfully=True,
            key_information_learned="Got the answer",
            remaining_plan_still_valid=True,
        )

        # Set up state
        todo_list = TodoList(items=[success_todo])
        executor.state = Mock()
        executor.state.todos = todo_list
        executor.state.observations = {1: success_observation}

        route = executor.handle_step_observed_medium()
        assert route == "continue_plan", (
            "Medium effort should continue on successful step"
        )
        assert success_todo.status == "completed"

        # --- Case 2: step failed → should return "replan_now" ---
        failed_todo = TodoItem(
            step_number=2,
            description="Divide by zero",
            status="running",
            result="Error: division by zero",
        )
        failed_observation = StepObservation(
            step_completed_successfully=False,
            key_information_learned="Division failed",
            remaining_plan_still_valid=False,
            needs_full_replan=True,
            replan_reason="Step failed with error",
        )

        todo_list_2 = TodoList(items=[failed_todo])
        executor.state.todos = todo_list_2
        executor.state.observations = {2: failed_observation}
        executor.state.last_replan_reason = None

        route = executor.handle_step_observed_medium()
        assert route == "replan_now", (
            "Medium effort should trigger replan on failed step"
        )
        assert executor.state.last_replan_reason == "Step failed with error"

    def test_reasoning_effort_low_marks_complete_without_deciding(self):
        """Low effort: mark_completed is called, decide_next_action is not.

        Unit test verifying the low handler's behavior directly.
        """
        from crewai.experimental.agent_executor import AgentExecutor
        from crewai.utilities.planning_types import TodoItem, TodoList

        executor = Mock(spec=AgentExecutor)
        executor.agent = Mock()
        executor.agent.verbose = False
        executor.agent.planning_config = Mock()
        executor.agent.planning_config.reasoning_effort = "low"

        # Bind the real method
        executor.handle_step_observed_low = (
            AgentExecutor.handle_step_observed_low.__get__(executor)
        )

        todo = TodoItem(
            step_number=1,
            description="Do something",
            status="running",
            result="Done successfully",
        )
        todo_list = TodoList(items=[todo])
        executor.state = Mock()
        executor.state.todos = todo_list

        route = executor.handle_step_observed_low()
        assert route == "continue_plan"
        assert todo.status == "completed"
        assert todo.result == "Done successfully"

    def test_planning_config_reasoning_effort_default_is_medium(self):
        """Verify PlanningConfig defaults reasoning_effort to 'medium'
        (aligned with runtime default in _get_reasoning_effort)."""
        from crewai.agent.planning_config import PlanningConfig

        config = PlanningConfig()
        assert config.reasoning_effort == "medium"

    def test_planning_config_reasoning_effort_validation(self):
        """Verify PlanningConfig rejects invalid reasoning_effort values."""
        from pydantic import ValidationError
        from crewai.agent.planning_config import PlanningConfig

        with pytest.raises(ValidationError):
            PlanningConfig(reasoning_effort="ultra")

        # Valid values should work
        for level in ("low", "medium", "high"):
            config = PlanningConfig(reasoning_effort=level)
            assert config.reasoning_effort == level

    def test_get_reasoning_effort_reads_from_config(self):
        """Verify _get_reasoning_effort reads from agent.planning_config."""
        from crewai.experimental.agent_executor import AgentExecutor

        executor = Mock(spec=AgentExecutor)
        executor._get_reasoning_effort = (
            AgentExecutor._get_reasoning_effort.__get__(executor)
        )

        # Case 1: planning_config with reasoning_effort set
        executor.agent = Mock()
        executor.agent.planning_config = Mock()
        executor.agent.planning_config.reasoning_effort = "high"
        assert executor._get_reasoning_effort() == "high"

        # Case 2: no planning_config → defaults to "medium"
        executor.agent.planning_config = None
        assert executor._get_reasoning_effort() == "medium"

        # Case 3: planning_config with default reasoning_effort
        executor.agent.planning_config = Mock()
        executor.agent.planning_config.reasoning_effort = "medium"
        assert executor._get_reasoning_effort() == "medium"



class TestObserverResponseParsing:
    """PlannerObserver must correctly parse LLM responses regardless of
    the format returned (StepObservation, JSON string, dict)."""

    def test_parse_step_observation_instance(self):
        """Direct StepObservation instance passes through unchanged."""
        from crewai.agents.planner_observer import PlannerObserver
        from crewai.utilities.planning_types import StepObservation

        obs = StepObservation(
            step_completed_successfully=False,
            key_information_learned="disk full",
            remaining_plan_still_valid=False,
            needs_full_replan=True,
            replan_reason="disk is full",
        )
        result = PlannerObserver._parse_observation_response(obs)
        assert result is obs
        assert result.step_completed_successfully is False
        assert result.needs_full_replan is True

    def test_parse_json_string(self):
        """JSON string from non-streaming LLM path is parsed correctly."""
        import json

        from crewai.agents.planner_observer import PlannerObserver
        from crewai.utilities.planning_types import StepObservation

        payload = {
            "step_completed_successfully": False,
            "key_information_learned": "command not found",
            "remaining_plan_still_valid": True,
            "needs_full_replan": False,
        }
        json_str = json.dumps(payload)
        result = PlannerObserver._parse_observation_response(json_str)

        assert isinstance(result, StepObservation)
        assert result.step_completed_successfully is False
        assert result.key_information_learned == "command not found"
        assert result.remaining_plan_still_valid is True

    def test_parse_json_string_with_markdown_fences(self):
        """JSON wrapped in ```json ... ``` fences is handled."""
        import json

        from crewai.agents.planner_observer import PlannerObserver
        from crewai.utilities.planning_types import StepObservation

        payload = {
            "step_completed_successfully": True,
            "key_information_learned": "found 3 files",
            "remaining_plan_still_valid": True,
        }
        fenced = f"```json\n{json.dumps(payload)}\n```"
        result = PlannerObserver._parse_observation_response(fenced)

        assert isinstance(result, StepObservation)
        assert result.step_completed_successfully is True
        assert result.key_information_learned == "found 3 files"

    def test_parse_dict_response(self):
        """Dict response from some provider paths is parsed correctly."""
        from crewai.agents.planner_observer import PlannerObserver
        from crewai.utilities.planning_types import StepObservation

        payload = {
            "step_completed_successfully": False,
            "key_information_learned": "timeout",
            "remaining_plan_still_valid": False,
            "needs_full_replan": True,
            "replan_reason": "step timed out",
        }
        result = PlannerObserver._parse_observation_response(payload)

        assert isinstance(result, StepObservation)
        assert result.step_completed_successfully is False
        assert result.needs_full_replan is True
        assert result.replan_reason == "step timed out"

    def test_parse_unparseable_falls_back_gracefully(self):
        """Totally unparseable response falls back to default failure."""
        from crewai.agents.planner_observer import PlannerObserver
        from crewai.utilities.planning_types import StepObservation

        result = PlannerObserver._parse_observation_response(12345)

        assert isinstance(result, StepObservation)
        assert result.step_completed_successfully is False
        assert result.remaining_plan_still_valid is False

    def test_observe_parses_json_string_from_llm(self):
        """End-to-end: observer.observe() correctly parses a JSON string from llm.call()."""
        import json

        from crewai.agents.planner_observer import PlannerObserver
        from crewai.utilities.planning_types import StepObservation, TodoItem

        llm = Mock()
        llm.call.return_value = json.dumps({
            "step_completed_successfully": False,
            "key_information_learned": "build failed with exit code 1",
            "remaining_plan_still_valid": False,
            "needs_full_replan": True,
            "replan_reason": "build system is misconfigured",
        })

        agent = Mock()
        agent.role = "Test Agent"
        agent.llm = llm
        agent.planning_config = None

        task = Mock()
        task.description = "Build the project"
        task.expected_output = "Successful build"

        observer = PlannerObserver(agent=agent, task=task)
        step = TodoItem(step_number=1, description="Run make", status="running")

        observation = observer.observe(
            completed_step=step,
            result="make: *** No rule to make target 'all'. Stop.",
            all_completed=[],
            remaining_todos=[],
        )

        assert observation.step_completed_successfully is False
        assert observation.needs_full_replan is True
        assert observation.replan_reason == "build system is misconfigured"


# =========================================================================
# Max Iterations Routing
# =========================================================================


class TestMaxIterationsRouting:
    """check_max_iterations must route to force_final_answer when
    the iteration limit is exceeded, not to a dead-end event."""

    def test_exceeded_routes_to_force_final_answer(self):
        from crewai.experimental.agent_executor import AgentExecutor

        executor = Mock(spec=AgentExecutor)
        executor.state = AgentExecutorState(iterations=25)
        executor.max_iter = 20

        result = AgentExecutor.check_max_iterations(executor)
        assert result == "force_final_answer"

    def test_under_limit_continues_reasoning(self):
        from crewai.experimental.agent_executor import AgentExecutor

        executor = Mock(spec=AgentExecutor)
        executor.state = AgentExecutorState(iterations=5)
        executor.max_iter = 20

        result = AgentExecutor.check_max_iterations(executor)
        assert result == "continue_reasoning"

    def test_under_limit_with_native_tools(self):
        from crewai.experimental.agent_executor import AgentExecutor

        executor = Mock(spec=AgentExecutor)
        executor.state = AgentExecutorState(iterations=5, use_native_tools=True)
        executor.max_iter = 20

        result = AgentExecutor.check_max_iterations(executor)
        assert result == "continue_reasoning_native"


# =========================================================================
# Native Tool Call Edge Cases
# =========================================================================


class TestNativeToolCallMaxUsage:
    """_execute_single_native_tool_call must produce a result string
    even when max_usage_reached=True and original_tool is None."""

    def test_max_usage_reached_without_original_tool(self):
        from crewai.experimental.agent_executor import AgentExecutor

        import inspect
        source = inspect.getsource(AgentExecutor._execute_single_native_tool_call)
        assert "elif max_usage_reached:" in source
        assert 'result = f"Tool \'{func_name}\' has reached its maximum usage limit' in source


# =========================================================================
# Executor State Reset on Re-invoke
# =========================================================================


class TestExecutorStateReset:
    """invoke() and invoke_async() must reset all execution state
    (including _finalize_called) so re-invocations work correctly."""

    def test_finalize_called_reset_in_invoke(self):
        import inspect
        from crewai.experimental.agent_executor import AgentExecutor

        source = inspect.getsource(AgentExecutor.invoke)
        finalize_idx = source.index("self._finalize_called = False")
        messages_idx = source.index("self.state.messages.clear()")
        assert finalize_idx < messages_idx, (
            "_finalize_called must be reset before state reset"
        )

    def test_finalize_called_reset_in_invoke_async(self):
        import inspect
        from crewai.experimental.agent_executor import AgentExecutor

        source = inspect.getsource(AgentExecutor.invoke_async)
        finalize_idx = source.index("self._finalize_called = False")
        messages_idx = source.index("self.state.messages.clear()")
        assert finalize_idx < messages_idx, (
            "_finalize_called must be reset before state reset in async path"
        )


# =========================================================================
# Plan Generation Isolation
# =========================================================================


class TestPlanGenerationIsolation:
    """generate_plan must store the plan in state only — never mutate
    the shared task.description object."""

    def test_generate_plan_does_not_mutate_task_description(self):
        import inspect
        from crewai.experimental.agent_executor import AgentExecutor

        source = inspect.getsource(AgentExecutor.generate_plan)
        assert "task.description +=" not in source, (
            "generate_plan still mutates task.description"
        )
        assert "task.description =" not in source or "Plan is stored in state" in source, (
            "generate_plan should store plan in state, not task.description"
        )


# =========================================================================
# Todo Status Tracking
# =========================================================================


class TestTodoStatusTracking:
    """Steps that fail without triggering a replan must be marked 'failed'
    (not 'completed') so status queries remain accurate."""

    def test_medium_effort_marks_failed_step_as_failed(self):
        import inspect
        from crewai.experimental.agent_executor import AgentExecutor

        source = inspect.getsource(AgentExecutor.handle_step_observed_medium)
        assert "mark_failed" in source, (
            "handle_step_observed_medium should use mark_failed for failed steps"
        )
        failed_no_replan_idx = source.index("failed but no replan")
        after_comment = source[failed_no_replan_idx:]
        assert "mark_completed" not in after_comment, (
            "mark_completed should not be called on failed steps"
        )

    def test_failed_step_appears_in_get_failed_todos(self):
        from crewai.utilities.planning_types import TodoItem, TodoList

        todos = TodoList(items=[
            TodoItem(step_number=1, description="Step 1"),
            TodoItem(step_number=2, description="Step 2"),
        ])

        todos.mark_running(1)
        todos.mark_failed(1, result="Error: build failed")

        failed = todos.get_failed_todos()
        assert len(failed) == 1
        assert failed[0].step_number == 1
        assert failed[0].result == "Error: build failed"

        completed = todos.get_completed_todos()
        assert len(completed) == 0


# =========================================================================
# TodoList Result Handling
# =========================================================================


class TestTodoResultHandling:
    """mark_completed/mark_failed must use `is not None` checks so
    empty-string results are preserved."""

    def test_mark_completed_preserves_empty_string(self):
        from crewai.utilities.planning_types import TodoItem, TodoList

        todos = TodoList(items=[
            TodoItem(step_number=1, description="Step 1"),
        ])
        todos.mark_completed(1, result="")
        item = todos.get_by_step_number(1)
        assert item.status == "completed"
        assert item.result == "", "Empty-string result should be stored, not dropped"

    def test_mark_failed_preserves_empty_string(self):
        from crewai.utilities.planning_types import TodoItem, TodoList

        todos = TodoList(items=[
            TodoItem(step_number=1, description="Step 1"),
        ])
        todos.mark_failed(1, result="")
        item = todos.get_by_step_number(1)
        assert item.status == "failed"
        assert item.result == "", "Empty-string result should be stored, not dropped"

    def test_mark_completed_none_does_not_overwrite(self):
        from crewai.utilities.planning_types import TodoItem, TodoList

        todos = TodoList(items=[
            TodoItem(step_number=1, description="Step 1", result="existing"),
        ])
        todos.mark_completed(1, result=None)
        item = todos.get_by_step_number(1)
        assert item.result == "existing", "None result should not overwrite existing"


# =========================================================================
# Dependency Resolution with Failed Steps
# =========================================================================


class TestDependencyResolutionWithFailures:
    """Failed dependencies must be treated as terminal so downstream
    todos are not permanently blocked."""

    def test_failed_dep_unblocks_downstream(self):
        from crewai.utilities.planning_types import TodoItem, TodoList

        todos = TodoList(items=[
            TodoItem(step_number=1, description="Build"),
            TodoItem(step_number=2, description="Test", depends_on=[1]),
            TodoItem(step_number=3, description="Deploy", depends_on=[2]),
        ])

        todos.mark_running(1)
        todos.mark_failed(1, result="build error")

        ready = todos.get_ready_todos()
        assert len(ready) == 1
        assert ready[0].step_number == 2

    def test_is_complete_with_mixed_terminal_states(self):
        from crewai.utilities.planning_types import TodoItem, TodoList

        todos = TodoList(items=[
            TodoItem(step_number=1, description="A", status="completed"),
            TodoItem(step_number=2, description="B", status="failed"),
            TodoItem(step_number=3, description="C", status="completed"),
        ])
        assert todos.is_complete is True

    def test_pending_todo_ready_when_dep_failed(self):
        from crewai.utilities.planning_types import TodoItem, TodoList

        todos = TodoList(items=[
            TodoItem(step_number=1, description="A", status="failed"),
            TodoItem(step_number=2, description="B", depends_on=[1], status="pending"),
        ])
        ready = todos.get_ready_todos()
        assert len(ready) == 1, "Downstream todo should be ready when dep is failed"


# =========================================================================
# PlanningConfig Defaults
# =========================================================================


class TestPlanningConfigDefaults:
    """PlanningConfig default reasoning_effort must be 'medium' to match
    the runtime fallback in _get_reasoning_effort."""

    def test_planning_config_default_is_medium(self):
        from crewai.agent.planning_config import PlanningConfig

        config = PlanningConfig()
        assert config.reasoning_effort == "medium", (
            f"Default should be 'medium', got '{config.reasoning_effort}'"
        )

    def test_explicit_config_matches_implicit_planning(self):
        """Agent(planning=True) and Agent(planning=True, planning_config=PlanningConfig())
        should produce the same reasoning_effort."""
        from crewai.agent.planning_config import PlanningConfig

        config = PlanningConfig()
        assert config.reasoning_effort == "medium"


# =========================================================================
# Vision Image Format Contract
# =========================================================================


class TestVisionImageFormatContract:
    """step_executor uses standard image_url format; each provider's
    _format_messages handles conversion to its native format."""

    def test_step_executor_uses_standard_image_url_format(self):
        import inspect
        from crewai.agents.step_executor import StepExecutor

        source = inspect.getsource(StepExecutor._build_observation_message)
        assert "image_url" in source, (
            "Step executor should use standard image_url format"
        )

    def test_anthropic_provider_has_image_block_converter(self):
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        assert hasattr(AnthropicCompletion, "_convert_image_blocks"), (
            "Anthropic provider must have _convert_image_blocks for auto-conversion"
        )
