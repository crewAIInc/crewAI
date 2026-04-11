"""Tests for async agent executor functionality."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from crewai.agent import Agent
from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentAction, AgentFinish
from crewai.agents.tools_handler import ToolsHandler
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task
from crewai.tools.tool_types import ToolResult


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    llm = MagicMock(spec=BaseLLM)
    llm.supports_stop_words.return_value = True
    llm.stop = []
    return llm


@pytest.fixture
def test_agent(mock_llm: MagicMock) -> Agent:
    """Create a real Agent for testing."""
    return Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        llm=mock_llm,
        verbose=False,
    )


@pytest.fixture
def test_task(test_agent: Agent) -> Task:
    """Create a real Task for testing."""
    return Task(
        description="Test task description",
        expected_output="Test output",
        agent=test_agent,
    )


@pytest.fixture
def mock_tools_handler() -> MagicMock:
    """Create a mock tools handler."""
    return MagicMock(spec=ToolsHandler)


@pytest.fixture
def executor(
    mock_llm: MagicMock,
    test_agent: Agent,
    test_task: Task,
    mock_tools_handler: MagicMock,
) -> CrewAgentExecutor:
    """Create a CrewAgentExecutor instance for testing."""
    return CrewAgentExecutor(
        llm=mock_llm,
        task=test_task,
        crew=None,
        agent=test_agent,
        prompt={"prompt": "Test prompt {input} {tool_names} {tools}"},
        max_iter=5,
        tools=[],
        tools_names="",
        stop_words=["Observation:"],
        tools_description="",
        tools_handler=mock_tools_handler,
    )


class TestAsyncAgentExecutor:
    """Tests for async agent executor methods."""

    @pytest.mark.asyncio
    async def test_ainvoke_returns_output(self, executor: CrewAgentExecutor) -> None:
        """Test that ainvoke returns the expected output."""
        expected_output = "Final answer from agent"

        with patch.object(
            executor,
            "_ainvoke_loop",
            new_callable=AsyncMock,
            return_value=AgentFinish(
                thought="Done", output=expected_output, text="Final Answer: Done"
            ),
        ):
            with patch.object(executor, "_show_start_logs"):
                with patch.object(executor, "_save_to_memory"):
                    result = await executor.ainvoke(
                        {
                            "input": "test input",
                            "tool_names": "",
                            "tools": "",
                        }
                    )

        assert result == {"output": expected_output}

    @pytest.mark.asyncio
    async def test_ainvoke_loop_calls_aget_llm_response(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that _ainvoke_loop calls aget_llm_response."""
        with patch(
            "crewai.agents.crew_agent_executor.aget_llm_response",
            new_callable=AsyncMock,
            return_value="Thought: I know the answer\nFinal Answer: Test result",
        ) as mock_aget_llm:
            with patch.object(executor, "_show_logs"):
                result = await executor._ainvoke_loop()

        mock_aget_llm.assert_called_once()
        assert isinstance(result, AgentFinish)

    @pytest.mark.asyncio
    async def test_ainvoke_loop_handles_tool_execution(
        self,
        executor: CrewAgentExecutor,
    ) -> None:
        """Test that _ainvoke_loop handles tool execution asynchronously."""
        call_count = 0

        async def mock_llm_response(*args: Any, **kwargs: Any) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return (
                    "Thought: I need to use a tool\n"
                    "Action: test_tool\n"
                    'Action Input: {"arg": "value"}'
                )
            return "Thought: I have the answer\nFinal Answer: Tool result processed"

        with patch(
            "crewai.agents.crew_agent_executor.aget_llm_response",
            new_callable=AsyncMock,
            side_effect=mock_llm_response,
        ):
            with patch(
                "crewai.agents.crew_agent_executor.aexecute_tool_and_check_finality",
                new_callable=AsyncMock,
                return_value=ToolResult(result="Tool executed", result_as_answer=False),
            ) as mock_tool_exec:
                with patch.object(executor, "_show_logs"):
                    with patch.object(executor, "_handle_agent_action") as mock_handle:
                        mock_handle.return_value = AgentAction(
                            text="Tool result",
                            tool="test_tool",
                            tool_input='{"arg": "value"}',
                            thought="Used tool",
                            result="Tool executed",
                        )
                        result = await executor._ainvoke_loop()

        assert mock_tool_exec.called
        assert isinstance(result, AgentFinish)

    @pytest.mark.asyncio
    async def test_ainvoke_loop_respects_max_iterations(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that _ainvoke_loop respects max iterations."""
        executor.max_iter = 2

        async def always_return_action(*args: Any, **kwargs: Any) -> str:
            return (
                "Thought: I need to think more\n"
                "Action: some_tool\n"
                "Action Input: {}"
            )

        with patch(
            "crewai.agents.crew_agent_executor.aget_llm_response",
            new_callable=AsyncMock,
            side_effect=always_return_action,
        ):
            with patch(
                "crewai.agents.crew_agent_executor.aexecute_tool_and_check_finality",
                new_callable=AsyncMock,
                return_value=ToolResult(result="Tool result", result_as_answer=False),
            ):
                with patch(
                    "crewai.agents.crew_agent_executor.handle_max_iterations_exceeded",
                    return_value=AgentFinish(
                        thought="Max iterations",
                        output="Forced answer",
                        text="Max iterations reached",
                    ),
                ) as mock_max_iter:
                    with patch.object(executor, "_show_logs"):
                        with patch.object(executor, "_handle_agent_action") as mock_ha:
                            mock_ha.return_value = AgentAction(
                                text="Action",
                                tool="some_tool",
                                tool_input="{}",
                                thought="Thinking",
                            )
                            result = await executor._ainvoke_loop()

        mock_max_iter.assert_called_once()
        assert isinstance(result, AgentFinish)

    @pytest.mark.asyncio
    async def test_ainvoke_handles_exceptions(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that ainvoke properly propagates exceptions."""
        with patch.object(executor, "_show_start_logs"):
            with patch.object(
                executor,
                "_ainvoke_loop",
                new_callable=AsyncMock,
                side_effect=ValueError("Test error"),
            ):
                with pytest.raises(ValueError, match="Test error"):
                    await executor.ainvoke(
                        {"input": "test", "tool_names": "", "tools": ""}
                    )

    @pytest.mark.asyncio
    async def test_concurrent_ainvoke_calls(
        self, mock_llm: MagicMock, test_agent: Agent, test_task: Task,
        mock_tools_handler: MagicMock,
    ) -> None:
        """Test that multiple ainvoke calls can run concurrently."""
        max_concurrent = 0
        current_concurrent = 0
        lock = asyncio.Lock()

        async def create_and_run_executor(executor_id: int) -> dict[str, Any]:
            nonlocal max_concurrent, current_concurrent

            executor = CrewAgentExecutor(
                llm=mock_llm,
                task=test_task,
                crew=None,
                agent=test_agent,
                prompt={"prompt": "Test {input} {tool_names} {tools}"},
                max_iter=5,
                tools=[],
                tools_names="",
                stop_words=["Observation:"],
                tools_description="",
                tools_handler=mock_tools_handler,
            )

            async def delayed_response(*args: Any, **kwargs: Any) -> str:
                nonlocal max_concurrent, current_concurrent
                async with lock:
                    current_concurrent += 1
                    max_concurrent = max(max_concurrent, current_concurrent)
                await asyncio.sleep(0.01)
                async with lock:
                    current_concurrent -= 1
                return f"Thought: Done\nFinal Answer: Result from executor {executor_id}"

            with patch(
                "crewai.agents.crew_agent_executor.aget_llm_response",
                new_callable=AsyncMock,
                side_effect=delayed_response,
            ):
                with patch.object(executor, "_show_start_logs"):
                    with patch.object(executor, "_show_logs"):
                        with patch.object(executor, "_save_to_memory"):
                            return await executor.ainvoke(
                                {
                                    "input": f"test {executor_id}",
                                    "tool_names": "",
                                    "tools": "",
                                }
                            )

        results = await asyncio.gather(
            create_and_run_executor(1),
            create_and_run_executor(2),
            create_and_run_executor(3),
        )

        assert len(results) == 3
        assert all("output" in r for r in results)
        assert max_concurrent > 1, f"Expected concurrent execution, max concurrent was {max_concurrent}"


class TestInvokeStepCallback:
    """Tests for _invoke_step_callback with sync and async callbacks."""

    def test_invoke_step_callback_with_sync_callback(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that a sync step callback is called normally."""
        callback = Mock()
        executor.step_callback = callback
        answer = AgentFinish(thought="thinking", output="test", text="final")

        executor._invoke_step_callback(answer)

        callback.assert_called_once_with(answer)

    def test_invoke_step_callback_with_async_callback(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that an async step callback is awaited via asyncio.run."""
        async_callback = AsyncMock()
        executor.step_callback = async_callback
        answer = AgentFinish(thought="thinking", output="test", text="final")

        with patch("crewai.agents.crew_agent_executor.asyncio.run") as mock_run:
            executor._invoke_step_callback(answer)

            async_callback.assert_called_once_with(answer)
            mock_run.assert_called_once()

    def test_invoke_step_callback_with_none(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that no error is raised when step_callback is None."""
        executor.step_callback = None
        answer = AgentFinish(thought="thinking", output="test", text="final")

        # Should not raise
        executor._invoke_step_callback(answer)


class TestParseNativeToolCall:
    """Tests for _parse_native_tool_call covering multiple provider formats.

    Regression tests for issue #4748: Bedrock tool calls with 'input' field
    were returning empty arguments because the old code used
    ``func_info.get("arguments", "{}")`` which always returns a truthy
    default, preventing the ``or`` fallback to ``tool_call.get("input")``.
    """

    def test_openai_dict_format(self, executor: CrewAgentExecutor) -> None:
        """Test OpenAI dict format with function.arguments."""
        tool_call = {
            "id": "call_abc123",
            "function": {
                "name": "search_tool",
                "arguments": '{"query": "test"}',
            },
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert call_id == "call_abc123"
        assert func_name == "search_tool"
        assert func_args == '{"query": "test"}'

    def test_bedrock_dict_format_extracts_input(self, executor: CrewAgentExecutor) -> None:
        """Test AWS Bedrock dict format extracts 'input' field for arguments."""
        tool_call = {
            "toolUseId": "tooluse_xyz789",
            "name": "search_tool",
            "input": {"query": "AWS Bedrock features"},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        call_id, func_name, func_args = result
        assert call_id == "tooluse_xyz789"
        assert func_name == "search_tool"
        assert func_args == {"query": "AWS Bedrock features"}

    def test_bedrock_dict_format_not_empty_regression(self, executor: CrewAgentExecutor) -> None:
        """Regression test for #4748: Bedrock args must NOT be empty.

        Before the fix, ``func_info.get("arguments", "{}")`` returned the
        truthy string ``"{}"`` which short-circuited the ``or`` operator,
        so ``tool_call.get("input", {})`` was never evaluated.
        """
        tool_call = {
            "toolUseId": "tooluse_regression",
            "name": "search_tool",
            "input": {"query": "important query", "limit": 5},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        assert func_args == {"query": "important query", "limit": 5}
        assert func_args != {}
        assert func_args != "{}"

    def test_dict_without_function_or_input_returns_empty(self, executor: CrewAgentExecutor) -> None:
        """Test dict format with neither function.arguments nor input."""
        tool_call = {
            "id": "call_noop",
            "name": "no_args_tool",
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, func_name, func_args = result
        assert func_name == "no_args_tool"
        assert func_args == {}

    def test_openai_arguments_preferred_over_input(self, executor: CrewAgentExecutor) -> None:
        """Test that function.arguments takes precedence over input."""
        tool_call = {
            "id": "call_both",
            "function": {
                "name": "dual_tool",
                "arguments": '{"from": "openai"}',
            },
            "input": {"from": "bedrock"},
        }
        result = executor._parse_native_tool_call(tool_call)
        assert result is not None
        _, _, func_args = result
        assert func_args == '{"from": "openai"}'

    def test_returns_none_for_unrecognized(self, executor: CrewAgentExecutor) -> None:
        """Test that None is returned for unrecognized tool call formats."""
        result = executor._parse_native_tool_call(12345)
        assert result is None


class TestAsyncLLMResponseHelper:
    """Tests for aget_llm_response helper function."""

    @pytest.mark.asyncio
    async def test_aget_llm_response_calls_acall(self) -> None:
        """Test that aget_llm_response calls llm.acall."""
        from crewai.utilities.agent_utils import aget_llm_response
        from crewai.utilities.printer import Printer

        mock_llm = MagicMock()
        mock_llm.acall = AsyncMock(return_value="LLM response")

        result = await aget_llm_response(
            llm=mock_llm,
            messages=[{"role": "user", "content": "test"}],
            callbacks=[],
            printer=Printer(),
        )

        mock_llm.acall.assert_called_once()
        assert result == "LLM response"

    @pytest.mark.asyncio
    async def test_aget_llm_response_raises_on_empty_response(self) -> None:
        """Test that aget_llm_response raises ValueError on empty response."""
        from crewai.utilities.agent_utils import aget_llm_response
        from crewai.utilities.printer import Printer

        mock_llm = MagicMock()
        mock_llm.acall = AsyncMock(return_value="")

        with pytest.raises(ValueError, match="Invalid response from LLM call"):
            await aget_llm_response(
                llm=mock_llm,
                messages=[{"role": "user", "content": "test"}],
                callbacks=[],
                printer=Printer(),
            )

    @pytest.mark.asyncio
    async def test_aget_llm_response_propagates_exceptions(self) -> None:
        """Test that aget_llm_response propagates LLM exceptions."""
        from crewai.utilities.agent_utils import aget_llm_response
        from crewai.utilities.printer import Printer

        mock_llm = MagicMock()
        mock_llm.acall = AsyncMock(side_effect=RuntimeError("LLM error"))

        with pytest.raises(RuntimeError, match="LLM error"):
            await aget_llm_response(
                llm=mock_llm,
                messages=[{"role": "user", "content": "test"}],
                callbacks=[],
                printer=Printer(),
            )
