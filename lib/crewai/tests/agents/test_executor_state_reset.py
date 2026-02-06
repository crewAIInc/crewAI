"""Tests for CrewAgentExecutor state reset between task executions.

Verifies that invoke() and ainvoke() reset messages and iterations
before each execution, preventing state leakage across sequential tasks.
See: https://github.com/crewAIInc/crewAI/issues/4389
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.agents.crew_agent_executor import CrewAgentExecutor
from crewai.agents.parser import AgentFinish


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock()
    llm.supports_stop_words.return_value = True
    llm.stop = []
    return llm


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    agent.role = "Test Agent"
    agent.key = "test_agent_key"
    agent.verbose = False
    agent.id = "test_agent_id"
    return agent


@pytest.fixture
def mock_task() -> MagicMock:
    task = MagicMock()
    task.description = "Test task description"
    return task


@pytest.fixture
def mock_crew() -> MagicMock:
    crew = MagicMock()
    crew.verbose = False
    crew._train = False
    return crew


@pytest.fixture
def mock_tools_handler() -> MagicMock:
    return MagicMock()


@pytest.fixture
def executor(
    mock_llm: MagicMock,
    mock_agent: MagicMock,
    mock_task: MagicMock,
    mock_crew: MagicMock,
    mock_tools_handler: MagicMock,
) -> CrewAgentExecutor:
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


class TestCrewAgentExecutorStateReset:
    """Tests verifying state reset between task executions (issue #4389)."""

    def test_invoke_resets_messages(self, executor: CrewAgentExecutor) -> None:
        """Test that invoke() clears messages before each execution."""
        executor.messages = [
            {"role": "system", "content": "old system prompt"},
            {"role": "user", "content": "old task prompt"},
            {"role": "assistant", "content": "old response"},
        ]

        with patch.object(
            executor,
            "_invoke_loop",
            return_value=AgentFinish(
                thought="Done", output="result", text="Final Answer: result"
            ),
        ):
            with patch.object(executor, "_show_start_logs"):
                with patch.object(executor, "_create_short_term_memory"):
                    with patch.object(executor, "_create_long_term_memory"):
                        with patch.object(executor, "_create_external_memory"):
                            executor.invoke(
                                {"input": "new task", "tool_names": "", "tools": ""}
                            )

        assert not any(
            msg.get("content") == "old system prompt" for msg in executor.messages
        )
        assert not any(
            msg.get("content") == "old task prompt" for msg in executor.messages
        )
        assert not any(
            msg.get("content") == "old response" for msg in executor.messages
        )

    def test_invoke_resets_iterations(self, executor: CrewAgentExecutor) -> None:
        """Test that invoke() resets iterations to 0 before each execution."""
        executor.iterations = 7

        with patch.object(
            executor,
            "_invoke_loop",
            return_value=AgentFinish(
                thought="Done", output="result", text="Final Answer: result"
            ),
        ):
            with patch.object(executor, "_show_start_logs"):
                with patch.object(executor, "_create_short_term_memory"):
                    with patch.object(executor, "_create_long_term_memory"):
                        with patch.object(executor, "_create_external_memory"):
                            executor.invoke(
                                {"input": "test", "tool_names": "", "tools": ""}
                            )

        assert executor.iterations == 0

    def test_invoke_sequential_tasks_no_message_leakage(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that sequential invoke() calls don't leak messages between tasks."""
        invocation_messages: list[list[dict[str, Any]]] = []

        def capture_messages_invoke_loop() -> AgentFinish:
            invocation_messages.append(list(executor.messages))
            return AgentFinish(
                thought="Done", output="result", text="Final Answer: result"
            )

        with patch.object(
            executor, "_invoke_loop", side_effect=capture_messages_invoke_loop
        ):
            with patch.object(executor, "_show_start_logs"):
                with patch.object(executor, "_create_short_term_memory"):
                    with patch.object(executor, "_create_long_term_memory"):
                        with patch.object(executor, "_create_external_memory"):
                            executor.invoke(
                                {
                                    "input": "first task",
                                    "tool_names": "",
                                    "tools": "",
                                }
                            )

                            executor.messages.append(
                                {"role": "assistant", "content": "first task response"}
                            )
                            executor.iterations = 5

                            executor.invoke(
                                {
                                    "input": "second task",
                                    "tool_names": "",
                                    "tools": "",
                                }
                            )

        assert len(invocation_messages) == 2
        first_msgs = invocation_messages[0]
        second_msgs = invocation_messages[1]

        assert not any(
            "second task" in str(msg.get("content", "")) for msg in first_msgs
        )
        assert not any(
            "first task" in str(msg.get("content", "")) for msg in second_msgs
        )

    def test_invoke_sequential_tasks_iterations_reset(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that iterations are reset between sequential invoke() calls."""
        iterations_at_start: list[int] = []

        original_setup = executor._setup_messages

        def capture_iterations_setup(inputs: dict[str, Any]) -> None:
            iterations_at_start.append(executor.iterations)
            original_setup(inputs)

        with patch.object(
            executor, "_setup_messages", side_effect=capture_iterations_setup
        ):
            with patch.object(
                executor,
                "_invoke_loop",
                return_value=AgentFinish(
                    thought="Done", output="result", text="Final Answer: result"
                ),
            ):
                with patch.object(executor, "_show_start_logs"):
                    with patch.object(executor, "_create_short_term_memory"):
                        with patch.object(executor, "_create_long_term_memory"):
                            with patch.object(executor, "_create_external_memory"):
                                executor.invoke(
                                    {
                                        "input": "task 1",
                                        "tool_names": "",
                                        "tools": "",
                                    }
                                )
                                executor.iterations = 3

                                executor.invoke(
                                    {
                                        "input": "task 2",
                                        "tool_names": "",
                                        "tools": "",
                                    }
                                )

        assert iterations_at_start[0] == 0
        assert iterations_at_start[1] == 0

    @pytest.mark.asyncio
    async def test_ainvoke_resets_messages(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that ainvoke() clears messages before each execution."""
        executor.messages = [
            {"role": "system", "content": "old system prompt"},
            {"role": "user", "content": "old task prompt"},
            {"role": "assistant", "content": "old response"},
        ]

        with patch.object(
            executor,
            "_ainvoke_loop",
            new_callable=AsyncMock,
            return_value=AgentFinish(
                thought="Done", output="result", text="Final Answer: result"
            ),
        ):
            with patch.object(executor, "_show_start_logs"):
                with patch.object(executor, "_create_short_term_memory"):
                    with patch.object(executor, "_create_long_term_memory"):
                        with patch.object(executor, "_create_external_memory"):
                            await executor.ainvoke(
                                {"input": "new task", "tool_names": "", "tools": ""}
                            )

        assert not any(
            msg.get("content") == "old system prompt" for msg in executor.messages
        )
        assert not any(
            msg.get("content") == "old task prompt" for msg in executor.messages
        )
        assert not any(
            msg.get("content") == "old response" for msg in executor.messages
        )

    @pytest.mark.asyncio
    async def test_ainvoke_resets_iterations(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that ainvoke() resets iterations to 0 before each execution."""
        executor.iterations = 7

        with patch.object(
            executor,
            "_ainvoke_loop",
            new_callable=AsyncMock,
            return_value=AgentFinish(
                thought="Done", output="result", text="Final Answer: result"
            ),
        ):
            with patch.object(executor, "_show_start_logs"):
                with patch.object(executor, "_create_short_term_memory"):
                    with patch.object(executor, "_create_long_term_memory"):
                        with patch.object(executor, "_create_external_memory"):
                            await executor.ainvoke(
                                {"input": "test", "tool_names": "", "tools": ""}
                            )

        assert executor.iterations == 0

    @pytest.mark.asyncio
    async def test_ainvoke_sequential_tasks_no_message_leakage(
        self, executor: CrewAgentExecutor
    ) -> None:
        """Test that sequential ainvoke() calls don't leak messages between tasks."""
        invocation_messages: list[list[dict[str, Any]]] = []

        async def capture_messages_ainvoke_loop() -> AgentFinish:
            invocation_messages.append(list(executor.messages))
            return AgentFinish(
                thought="Done", output="result", text="Final Answer: result"
            )

        with patch.object(
            executor,
            "_ainvoke_loop",
            new_callable=AsyncMock,
            side_effect=capture_messages_ainvoke_loop,
        ):
            with patch.object(executor, "_show_start_logs"):
                with patch.object(executor, "_create_short_term_memory"):
                    with patch.object(executor, "_create_long_term_memory"):
                        with patch.object(executor, "_create_external_memory"):
                            await executor.ainvoke(
                                {
                                    "input": "first task",
                                    "tool_names": "",
                                    "tools": "",
                                }
                            )

                            executor.messages.append(
                                {"role": "assistant", "content": "first task response"}
                            )
                            executor.iterations = 5

                            await executor.ainvoke(
                                {
                                    "input": "second task",
                                    "tool_names": "",
                                    "tools": "",
                                }
                            )

        assert len(invocation_messages) == 2
        first_msgs = invocation_messages[0]
        second_msgs = invocation_messages[1]

        assert not any(
            "second task" in str(msg.get("content", "")) for msg in first_msgs
        )
        assert not any(
            "first task" in str(msg.get("content", "")) for msg in second_msgs
        )
