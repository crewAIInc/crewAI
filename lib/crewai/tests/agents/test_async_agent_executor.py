"""Tests for async agent executor functionality."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentAction, AgentFinish
from crewai.tools.tool_types import ToolResult


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    llm = MagicMock()
    llm.supports_stop_words.return_value = True
    llm.stop = []
    return llm


@pytest.fixture
def mock_agent() -> MagicMock:
    """Create a mock agent for testing."""
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.key = "test_agent_key"
    agent.verbose = False
    agent.id = "test_agent_id"
    return agent


@pytest.fixture
def mock_task() -> MagicMock:
    """Create a mock task for testing."""
    task = MagicMock()
    task.description = "Test task description"
    return task


@pytest.fixture
def mock_crew() -> MagicMock:
    """Create a mock crew for testing."""
    crew = MagicMock()
    crew.verbose = False
    crew._train = False
    return crew


@pytest.fixture
def mock_tools_handler() -> MagicMock:
    """Create a mock tools handler."""
    return MagicMock()


@pytest.fixture
def executor(
    mock_llm: MagicMock,
    mock_agent: MagicMock,
    mock_task: MagicMock,
    mock_crew: MagicMock,
    mock_tools_handler: MagicMock,
) -> CrewAgentExecutor:
    """Create a CrewAgentExecutor instance for testing."""
    return CrewAgentExecutor(
        llm=mock_llm,
        task=mock_task,
        crew=mock_crew,
        agent=mock_agent,
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
                with patch.object(executor, "_create_short_term_memory"):
                    with patch.object(executor, "_create_long_term_memory"):
                        with patch.object(executor, "_create_external_memory"):
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
        self, mock_llm: MagicMock, mock_agent: MagicMock, mock_task: MagicMock,
        mock_crew: MagicMock, mock_tools_handler: MagicMock
    ) -> None:
        """Test that multiple ainvoke calls can run concurrently."""

        async def create_and_run_executor(executor_id: int) -> dict[str, Any]:
            executor = CrewAgentExecutor(
                llm=mock_llm,
                task=mock_task,
                crew=mock_crew,
                agent=mock_agent,
                prompt={"prompt": "Test {input} {tool_names} {tools}"},
                max_iter=5,
                tools=[],
                tools_names="",
                stop_words=["Observation:"],
                tools_description="",
                tools_handler=mock_tools_handler,
            )

            async def delayed_response(*args: Any, **kwargs: Any) -> str:
                await asyncio.sleep(0.05)
                return f"Thought: Done\nFinal Answer: Result from executor {executor_id}"

            with patch(
                "crewai.agents.crew_agent_executor.aget_llm_response",
                new_callable=AsyncMock,
                side_effect=delayed_response,
            ):
                with patch.object(executor, "_show_start_logs"):
                    with patch.object(executor, "_show_logs"):
                        with patch.object(executor, "_create_short_term_memory"):
                            with patch.object(executor, "_create_long_term_memory"):
                                with patch.object(executor, "_create_external_memory"):
                                    return await executor.ainvoke(
                                        {
                                            "input": f"test {executor_id}",
                                            "tool_names": "",
                                            "tools": "",
                                        }
                                    )

        import time

        start = time.time()
        results = await asyncio.gather(
            create_and_run_executor(1),
            create_and_run_executor(2),
            create_and_run_executor(3),
        )
        elapsed = time.time() - start

        assert len(results) == 3
        assert all("output" in r for r in results)
        assert elapsed < 0.15, f"Expected concurrent execution, took {elapsed}s"


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