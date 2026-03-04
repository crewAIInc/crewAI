"""Tests for output_pydantic behavior in ReAct flow when LLM doesn't support function calling.

Regression tests for https://github.com/crewAIInc/crewAI/issues/4695

When an LLM does NOT support function calling (supports_function_calling() returns False),
the executor should use the ReAct text-based pattern. In this path, response_model should
NOT be passed to the LLM call, because doing so forces structured output (via instructor/
tools mode) before the agent can reason through the Action/Observation loop.

The schema should still be embedded in the prompt text for guidance, and the final
conversion to pydantic/json should happen in task._export_output() after the ReAct loop.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentFinish


# ---------------------------------------------------------------------------
# Pydantic models used as output_pydantic in tests
# ---------------------------------------------------------------------------


class PersonInfo(BaseModel):
    """A simple pydantic model for testing output_pydantic."""

    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")


class WeatherReport(BaseModel):
    """Another pydantic model for testing output_pydantic."""

    city: str = Field(description="City name")
    temperature: float = Field(description="Temperature in Fahrenheit")
    condition: str = Field(description="Weather condition")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_llm(*, supports_fc: bool) -> MagicMock:
    """Create a mock LLM with configurable function-calling support."""
    llm = MagicMock()
    llm.supports_function_calling.return_value = supports_fc
    llm.supports_stop_words.return_value = True
    llm.stop = []
    return llm


def _make_executor(
    llm: MagicMock,
    *,
    response_model: type[BaseModel] | None = None,
) -> CrewAgentExecutor:
    """Create a CrewAgentExecutor with the given LLM and response_model."""
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.key = "test_agent_key"
    agent.verbose = False
    agent.id = "test_agent_id"

    task = MagicMock()
    task.description = "Test task"

    crew = MagicMock()
    crew.verbose = False
    crew._train = False

    executor = CrewAgentExecutor(
        llm=llm,
        task=task,
        crew=crew,
        agent=agent,
        prompt={"prompt": "Test prompt {input} {tool_names} {tools}"},
        max_iter=5,
        tools=[],
        tools_names="",
        stop_words=["Observation:"],
        tools_description="",
        tools_handler=MagicMock(),
        response_model=response_model,
    )
    return executor


# ===========================================================================
# Sync tests
# ===========================================================================


class TestReActFlowDoesNotPassResponseModel:
    """Verify that _invoke_loop_react does NOT pass response_model to LLM."""

    def test_react_flow_passes_none_response_model_when_output_pydantic_set(
        self,
    ) -> None:
        """When output_pydantic is set but LLM lacks function calling,
        response_model must be None in the get_llm_response call."""
        llm = _make_llm(supports_fc=False)
        executor = _make_executor(llm, response_model=PersonInfo)

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            return_value="Thought: I know the answer\nFinal Answer: John is 30 years old",
        ) as mock_get_llm:
            with patch.object(executor, "_show_logs"):
                result = executor._invoke_loop()

        # The critical assertion: response_model must be None in ReAct flow
        call_kwargs = mock_get_llm.call_args
        assert call_kwargs.kwargs.get("response_model") is None, (
            "response_model should be None in ReAct flow, but got "
            f"{call_kwargs.kwargs.get('response_model')}"
        )
        assert isinstance(result, AgentFinish)

    def test_react_flow_does_not_use_instructor_for_non_fc_llm(self) -> None:
        """Ensure InternalInstructor is never invoked in the ReAct path."""
        llm = _make_llm(supports_fc=False)
        executor = _make_executor(llm, response_model=WeatherReport)

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            return_value="Thought: I found the weather\nFinal Answer: It is sunny in NYC at 72F",
        ):
            with patch.object(executor, "_show_logs"):
                with patch(
                    "crewai.utilities.internal_instructor.InternalInstructor"
                ) as mock_instructor:
                    executor._invoke_loop()

        mock_instructor.assert_not_called()

    def test_invoke_loop_routes_to_react_when_no_function_calling(self) -> None:
        """Confirm _invoke_loop routes to _invoke_loop_react when
        supports_function_calling() returns False."""
        llm = _make_llm(supports_fc=False)
        executor = _make_executor(llm, response_model=PersonInfo)

        with patch.object(
            executor,
            "_invoke_loop_react",
            return_value=AgentFinish(thought="done", output="test", text="Final Answer: test"),
        ) as mock_react:
            with patch.object(executor, "_invoke_loop_native_tools") as mock_native:
                executor._invoke_loop()

        mock_react.assert_called_once()
        mock_native.assert_not_called()

    def test_invoke_loop_routes_to_native_when_function_calling_supported(
        self,
    ) -> None:
        """Confirm _invoke_loop routes to _invoke_loop_native_tools when
        supports_function_calling() returns True AND tools are present."""
        llm = _make_llm(supports_fc=True)
        executor = _make_executor(llm, response_model=PersonInfo)
        # Need at least one tool for native path
        executor.original_tools = [MagicMock()]

        with patch.object(
            executor,
            "_invoke_loop_native_tools",
            return_value=AgentFinish(thought="done", output="test", text="Final Answer: test"),
        ) as mock_native:
            with patch.object(executor, "_invoke_loop_react") as mock_react:
                executor._invoke_loop()

        mock_native.assert_called_once()
        mock_react.assert_not_called()

    def test_react_flow_still_works_with_tool_usage(self) -> None:
        """Verify the ReAct loop still processes Action/Observation cycles
        correctly even when output_pydantic is set."""
        llm = _make_llm(supports_fc=False)
        executor = _make_executor(llm, response_model=PersonInfo)

        call_count = 0

        def mock_llm_response(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            # Verify response_model is None on every call
            assert kwargs.get("response_model") is None, (
                f"response_model should be None in ReAct flow (call {call_count})"
            )
            if call_count == 1:
                return (
                    "Thought: I need to search for the person\n"
                    "Action: search_tool\n"
                    'Action Input: {"query": "John Doe"}'
                )
            return (
                "Thought: I found the person info\n"
                "Final Answer: John Doe is 30 years old"
            )

        from crewai.tools.tool_types import ToolResult

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            side_effect=mock_llm_response,
        ):
            with patch(
                "crewai.agents.crew_agent_executor.execute_tool_and_check_finality",
                return_value=ToolResult(result="John Doe, age 30", result_as_answer=False),
            ):
                with patch.object(executor, "_show_logs"):
                    with patch.object(executor, "_handle_agent_action") as mock_handle:
                        from crewai.agents.parser import AgentAction

                        mock_handle.return_value = AgentAction(
                            text="Tool result",
                            tool="search_tool",
                            tool_input='{"query": "John Doe"}',
                            thought="Used tool",
                            result="John Doe, age 30",
                        )
                        result = executor._invoke_loop()

        assert isinstance(result, AgentFinish)
        assert call_count == 2, f"Expected 2 LLM calls, got {call_count}"

    def test_react_flow_without_response_model_unchanged(self) -> None:
        """Verify the ReAct flow still works normally when no response_model is set."""
        llm = _make_llm(supports_fc=False)
        executor = _make_executor(llm, response_model=None)

        with patch(
            "crewai.agents.crew_agent_executor.get_llm_response",
            return_value="Thought: Simple answer\nFinal Answer: Hello world",
        ) as mock_get_llm:
            with patch.object(executor, "_show_logs"):
                result = executor._invoke_loop()

        call_kwargs = mock_get_llm.call_args
        assert call_kwargs.kwargs.get("response_model") is None
        assert isinstance(result, AgentFinish)


# ===========================================================================
# Async tests
# ===========================================================================


class TestAsyncReActFlowDoesNotPassResponseModel:
    """Verify that _ainvoke_loop_react does NOT pass response_model to LLM."""

    @pytest.mark.asyncio
    async def test_async_react_flow_passes_none_response_model(self) -> None:
        """Async variant: response_model must be None in ReAct flow."""
        llm = _make_llm(supports_fc=False)
        executor = _make_executor(llm, response_model=PersonInfo)

        with patch(
            "crewai.agents.crew_agent_executor.aget_llm_response",
            new_callable=AsyncMock,
            return_value="Thought: I know\nFinal Answer: John is 30",
        ) as mock_aget_llm:
            with patch.object(executor, "_show_logs"):
                result = await executor._ainvoke_loop()

        call_kwargs = mock_aget_llm.call_args
        assert call_kwargs.kwargs.get("response_model") is None, (
            "response_model should be None in async ReAct flow"
        )
        assert isinstance(result, AgentFinish)

    @pytest.mark.asyncio
    async def test_async_invoke_loop_routes_to_react_when_no_fc(self) -> None:
        """Async: _ainvoke_loop routes to _ainvoke_loop_react when
        supports_function_calling() returns False."""
        llm = _make_llm(supports_fc=False)
        executor = _make_executor(llm, response_model=WeatherReport)

        with patch.object(
            executor,
            "_ainvoke_loop_react",
            new_callable=AsyncMock,
            return_value=AgentFinish(thought="done", output="test", text="Final Answer: test"),
        ) as mock_react:
            with patch.object(executor, "_ainvoke_loop_native_tools") as mock_native:
                await executor._ainvoke_loop()

        mock_react.assert_called_once()
        mock_native.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_react_flow_with_tool_usage(self) -> None:
        """Async: ReAct loop processes tool calls correctly with output_pydantic."""
        llm = _make_llm(supports_fc=False)
        executor = _make_executor(llm, response_model=PersonInfo)

        call_count = 0

        async def mock_llm_response(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            assert kwargs.get("response_model") is None
            if call_count == 1:
                return (
                    "Thought: I need to search\n"
                    "Action: search_tool\n"
                    'Action Input: {"query": "test"}'
                )
            return "Thought: Done\nFinal Answer: Result found"

        from crewai.tools.tool_types import ToolResult

        with patch(
            "crewai.agents.crew_agent_executor.aget_llm_response",
            new_callable=AsyncMock,
            side_effect=mock_llm_response,
        ):
            with patch(
                "crewai.agents.crew_agent_executor.aexecute_tool_and_check_finality",
                new_callable=AsyncMock,
                return_value=ToolResult(result="Found it", result_as_answer=False),
            ):
                with patch.object(executor, "_show_logs"):
                    with patch.object(executor, "_handle_agent_action") as mock_handle:
                        from crewai.agents.parser import AgentAction

                        mock_handle.return_value = AgentAction(
                            text="Tool result",
                            tool="search_tool",
                            tool_input='{"query": "test"}',
                            thought="Searching",
                            result="Found it",
                        )
                        result = await executor._ainvoke_loop()

        assert isinstance(result, AgentFinish)
        assert call_count == 2


# ===========================================================================
# Integration-style tests (Crew-level)
# ===========================================================================


class TestCrewLevelOutputPydanticWithNonFCModel:
    """Higher-level tests verifying that a Crew with output_pydantic works
    correctly when the LLM doesn't support function calling."""

    def test_crew_output_pydantic_with_non_fc_llm_uses_react(self) -> None:
        """A Crew with output_pydantic should still use ReAct flow and NOT
        pass response_model to the LLM when it doesn't support FC."""
        from crewai import Agent, Crew, Task

        llm = MagicMock()
        llm.supports_function_calling.return_value = False
        llm.supports_stop_words.return_value = True
        llm.stop = []
        llm.model = "ollama/llama3"
        # Return a valid ReAct final answer
        llm.call.return_value = (
            "Thought: I know the answer\n"
            'Final Answer: {"name": "John Doe", "age": 30}'
        )

        # Patch create_llm so Agent.__init__ doesn't try to instantiate a real LLM
        with patch("crewai.agent.core.create_llm", return_value=llm):
            agent = Agent(
                role="Researcher",
                goal="Find person info",
                backstory="You research people.",
                llm=llm,
                verbose=False,
            )

            task = Task(
                description="Find info about John Doe",
                expected_output="Person info as JSON",
                agent=agent,
                output_pydantic=PersonInfo,
            )

            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            result = crew.kickoff()

        # Verify llm.call was invoked
        assert llm.call.called

        # Verify response_model was NOT passed to llm.call
        for call_args in llm.call.call_args_list:
            rm = call_args.kwargs.get("response_model")
            assert rm is None, (
                f"response_model should be None for non-FC LLM, got {rm}"
            )

        assert result is not None

    def test_crew_output_pydantic_with_fc_llm_uses_native_tools(self) -> None:
        """A Crew with output_pydantic and an FC-capable LLM should use
        native tools flow and CAN pass response_model."""
        from crewai import Agent, Crew, Task

        llm = MagicMock()
        llm.supports_function_calling.return_value = True
        llm.supports_stop_words.return_value = True
        llm.stop = []
        llm.model = "gpt-4o-mini"
        # Return a valid final answer (no tool calls)
        llm.call.return_value = '{"name": "Jane Doe", "age": 25}'

        # Patch create_llm so Agent.__init__ doesn't try to instantiate a real LLM
        with patch("crewai.agent.core.create_llm", return_value=llm):
            agent = Agent(
                role="Researcher",
                goal="Find person info",
                backstory="You research people.",
                llm=llm,
                verbose=False,
            )

            task = Task(
                description="Find info about Jane Doe",
                expected_output="Person info as JSON",
                agent=agent,
                output_pydantic=PersonInfo,
            )

            crew = Crew(agents=[agent], tasks=[task], verbose=False)
            result = crew.kickoff()

        assert result is not None
